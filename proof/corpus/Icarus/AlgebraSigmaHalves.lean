/-
  `AlgebraSigmaHalves.lean` — does SHA-256's `Sigma0` / `Sigma1` / `sigma0` / `sigma1` compose
  under the **forced bit-halves pairing** of AR-17?

  Task AR-26 of the algebra-rooting wave. Owned files: this one, plus `AlgebraSigmaHalves.md`. No
  production, pin, extraction-root or API edit; no commit.

  The scope note that opens the task. AR-12 (`AlgebraSha256Sigma.lean`) refuted the family
  `x ↦ x·(a + r·b)` — a single `ZMod`-linear map of the word, read off the one-slot line
  `z = (x, 0)` — and from that refutation concluded that the composites are not carried. That
  conclusion was about the *one-slot* pairing, which is forced to place all `k` bits in one
  coordinate and is therefore `ZMod`-linear in `a`. AR-17 (`AlgebraRotorSha.lean`) then found a
  pairing that escapes the refutation: the word cut at `n` as
  `halves n a = (a mod 2^n, ⌊a/2^n⌋)`, read through the elliptic face by the multiplier
  `(2^{k−n}, −1)`, reproduces `rotr n` **exactly, for every amount and every width**. That
  pairing is not `ZMod`-linear in `a`; it is linear in the two *digit coordinates*.

  So AR-12's refutation does not apply to the composites under AR-17's pairing, and the question
  is open. This file answers it: per term, for the XOR layer, for the composite, and for cost.

  §1 The per-term reading. `halvesCell n a = ringMul (−1) (halves n a) (2^{k−n}, −1)`; its first
      slot is `rotrVal n a` with **no hypothesis on `n` or `a`** (`slot1_exact`), and its second
      slot is the named quantity `revHalves n a = ⌊a/2^n⌋·2^{k−n} − (a mod 2^n)`
      (`slot2_exact`, `revHalves_eq_sub`). The read is therefore exact per term with **no
      per-term residual**: AR-12's `rotResidual` and AR-13's correction step both disappear under
      the pairing. `rotrVal_eq_rotrBv` links `rotrVal` to the extracted `rotr` for `n < k`; at or
      above the width the pairing collapses onto the one-slot line and the read is the word
      itself (`halves_degenerate`, `halves_read_is_identity_above_width`), which is why the FIPS
      amounts need `k ≥ 26`. The forced multiplier is on the unit circle exactly when
      `2·(k−n) ≥ k` (`halves_unitary_of`, `halves_not_unitary_of`); at `k = 32` the FIPS
      rotations split, `2, 6, 7, 11, 13` unitary and `17, 18, 19, 22, 25` faithful off the circle
      (`sha256_amounts_unitary`).

  §2 The pairing is amount-specific. `Span2 k n g` is the class the pairing can read: the
      `ZMod`-linear functionals of the two digit coordinates. **`span_rotr_iff`:
      `Span2 k n (rotrVal k m) ↔ m = n`,** and `span_shr_iff` for the shift. The forced pairing
      is not a free re-encoding that any rotation may use: it serves exactly one amount. Both
      refutations use the witness `a = 2^m`, where the target value is `1` while both digit
      coordinates of `2^m` are even for every split `n ≠ m`. No conditionality on AR-23: the
      `SHR` statement needed here is proved here. Ground tables and ground refutations at `k = 4`
      and `k = 2` by `decide`.

  §3 The XOR layer does not compose, and the obstruction is in the **halves coordinates**, not in
      the word ring. `not_span_of_evenCell_oddVal` is the generic statement: an odd value at a
      word whose two digit coordinates are both even cannot be a linear functional of the cell.
      Instantiated at the four composites with their FIPS amounts, for every split `1 ≤ n < k` and
      every `k ≥ 26` (`bigSigma0_not_span`, `bigSigma1_not_span`, `smallSigma0_not_span`,
      `smallSigma1_not_span`), with the witness table of §3. What *is* proved is the composite as
      three halves-pairing reads plus the **named residual `zxor3`** and nothing else
      (`bigSigma0_reads` and siblings); `smallSigma*` pay **zero** ring operations for the `SHR`
      term, which is its own second coordinate (`shrRead_is_second_coordinate`).

  §4 The count of ring operations under the halves reading, against AR-15's round cost. The same
      model reproduces AR-15's published round vector exactly (`c5RoundOps_eq_ar15`), so the
      difference is a like-for-like difference.

  Trust boundary. Statements about `ZMod (2^k)` and `BitVec k`, and about re-declarations of the
  corpus bodies `Crypto.Sha2.{rotr,shrx,Sigma0,Sigma1,sigma0,sigma1}` with the width `32`
  replaced by a parameter. Nothing here reaches `rustc`/LLVM, the kernel ABI, GPU/SPIR-V or the
  `unsafe` container, and no security, collision-resistance or constant-time property is claimed
  or implied. No `decide` above `k = 4`; no `sorry`, no `native_decide`.
-/

import Apeiron.Trichotomy.AnyRing

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace AlgebraSigmaHalves

open Apeiron.Trichotomy.AnyRing

/-! ## §0 The carrier: the bit-halves pairing, the forced multiplier, the read -/

/-- The `k`-bit word of value `a` as an element of the ring `ZMod (2^k)`. -/
def denNat {k : Nat} (a : Nat) : ZMod (2 ^ k) := (a : Nat)

/-- The **low** digit of the forced pairing: the low `n` bits of the word. -/
def loBits (n a : Nat) : Nat := a % 2 ^ n

/-- The **high** digit of the forced pairing: the remaining `k − n` bits. -/
def hiBits (n a : Nat) : Nat := a / 2 ^ n

/-- **The forced bit-halves pairing** `(a mod 2^n, ⌊a/2^n⌋)`, as a two-slot point of
    `R[x]/(x² + 1)` over `ZMod (2^k)`. -/
def halves {k : Nat} (n a : Nat) : RingCell (ZMod (2 ^ k)) :=
  (denNat (loBits n a), denNat (hiBits n a))

/-- **The multiplier the pairing forces**: the low digit scaled by `2^{k−n}` — the lift of the
    wrapped bits — and the high digit by `−1`. -/
def halvesMul {k : Nat} (n : Nat) : RingCell (ZMod (2 ^ k)) := (denNat (2 ^ (k - n)), -1)

/-- The product cell: the word under the pairing, multiplied by the forced multiplier at the
    elliptic face `σ = −1`. -/
def halvesCell {k : Nat} (n a : Nat) : RingCell (ZMod (2 ^ k)) :=
  ringMul (-1) (halves (k := k) n a) (halvesMul (k := k) n)

/-- **The second-slot content, named** `revHalves n a = ⌊a/2^n⌋·2^{k−n} − (a mod 2^n)`: the high
    digit scaled by the gain exponent, minus the low digit. The read below never uses it; it is
    recorded because the task asks for each term's second coordinate, and because it is what
    makes the multiplier's determinant `2^{2(k−n)} + 1`. -/
def revHalves {k : Nat} (n a : Nat) : ZMod (2 ^ k) :=
  denNat (hiBits n a) * denNat (2 ^ (k - n)) - denNat (loBits n a)

/-- The read off the first slot at the elliptic face: `c·p − s·q`, the real part of the complex
    product `(p + qi)(c + si)`. -/
def ellipticRead {R : Type*} [CommRing R] (u z : RingCell R) : R := u.1 * z.1 - u.2 * z.2

/-- **One rotation term, read through the forced pairing**: the first slot of the forced product. -/
def rotrRead {k : Nat} (n a : Nat) : ZMod (2 ^ k) := (halvesCell (k := k) n a).1

/-- The `rotr n` value formula at width `k`: the logical right shift, plus the low `n` bits lifted
    to the top. -/
def rotrVal {k : Nat} (n a : Nat) : Nat := a / 2 ^ n + (a % 2 ^ n) * 2 ^ (k - n)

/-- The rotation as a `BitVec`: the corpus's `rotr` body at width `k`. -/
def rotrBv {k : Nat} (a n : Nat) : BitVec k := (BitVec.ofNat k a).rotateRight n

/-- The `SHR n` value at width `k`: the logical right shift. -/
def shrVal (n a : Nat) : Nat := a / 2 ^ n

/-- The shift as a `BitVec`: the corpus's `shrx` body at width `k`. -/
def shrxBv {k : Nat} (a n : Nat) : BitVec k := (BitVec.ofNat k a) >>> (n % k)

/-- **The exclusive-or of three words, on residues**: `⊕` through the representatives. This is
    the residual layer that §3 names. -/
def zxor3 {k : Nat} (a b c : ZMod (2 ^ k)) : ZMod (2 ^ k) :=
  denNat (k := k) (a.val ^^^ b.val ^^^ c.val)

/-! ## §1 The per-term reading, and the exact second-coordinate content -/

/-- The exact action of the face: `ringMul σ (p,q) (c,s) = (p·c + σ·q·s, p·s + q·c)`, no hypothesis
    on `σ`. Re-derived here, so that every count in §4 rests on this body. -/
theorem ringMul_cell_eq {R : Type*} [CommRing R] (σ p q c s : R) :
    ringMul σ (p, q) (c, s) = (p * c + σ * q * s, p * s + q * c) := by
  simp only [ringMul]

/-- The first slot of an elliptic product is `ellipticRead`. -/
theorem ellipticRead_eq {R : Type*} [CommRing R] (c s p q : R) :
    (ringMul (-1 : R) (p, q) (c, s)).1 = ellipticRead (c, s) (p, q) := by
  simp only [ellipticRead, ringMul]
  ring

/-- The `rotr n` value read into the ring, in the order the read produces it. -/
theorem denNat_rotrVal {k n a : Nat} :
    denNat (k := k) (rotrVal (k := k) n a)
      = denNat (k := k) (hiBits n a) + denNat (k := k) (loBits n a) * denNat (k := k) (2 ^ (k - n)) := by
  simp only [rotrVal, loBits, hiBits, denNat, Nat.cast_add, Nat.cast_mul]

/-- **PER TERM, THE READ: the forced pairing reproduces `rotr n` exactly, with no hypothesis on
    `n` or `a`.** The first slot is `(a mod 2^n)·2^{k−n} − (−1)·⌊a/2^n⌋`, i.e. `rotrVal n a`.

    This is the statement AR-12 could not make. It is *not* a `ZMod`-linear map of `a`, it is
    exact, and therefore there is **no per-term residual**: AR-12's `rotResidual` and AR-13's
    per-rotation quotient readout are both unnecessary under the pairing. -/
theorem slot1_exact (k n a : Nat) :
    rotrRead (k := k) n a = denNat (k := k) (rotrVal (k := k) n a) := by
  have key : rotrRead (k := k) n a
      = denNat (k := k) (loBits n a) * denNat (k := k) (2 ^ (k - n))
        + denNat (k := k) (hiBits n a) := by
    simp only [rotrRead, halvesCell, halves, halvesMul, ringMul, denNat]
    ring
  rw [key, denNat_rotrVal]
  simp only [denNat]
  ring

/-- The read, named: `rotrRead n a` is the `rotrVal n a` residue. Used by the composite forms of
    §3a, which need the value, not the read. -/
theorem rotrRead_eq_denNat (k n a : Nat) :
    rotrRead (k := k) n a = denNat (k := k) (rotrVal (k := k) n a) := slot1_exact k n a

/-- The same read through `ellipticRead`, the form AR-17 stated: the read of the forced
    multiplier on the paired cell is `rotrVal n a`. -/
theorem rotrRead_eq_ellipticRead (k n a : Nat) :
    ellipticRead (halvesMul (k := k) n) (halves (k := k) n a)
      = denNat (k := k) (rotrVal (k := k) n a) := by
  calc ellipticRead (halvesMul (k := k) n) (halves (k := k) n a)
      = (halvesCell (k := k) n a).1 := by
        simp only [halvesCell, halvesMul, halves]
        exact (ellipticRead_eq (c := denNat (k := k) (2 ^ (k - n))) (s := -1)
          (p := denNat (k := k) (loBits n a)) (q := denNat (k := k) (hiBits n a))).symm
    _ = denNat (k := k) (rotrVal (k := k) n a) := slot1_exact k n a

/-- **PER TERM, THE SECOND COORDINATE: its exact content is `revHalves n a`**, the high digit scaled
    by `2^{k−n}` minus the low digit, i.e. `−(a mod 2^n) + ⌊a/2^n⌋·2^{k−n}`. The read never uses
    it; it is recorded because the task asks for it, and because the two coordinates together are
    the invertible pair `(lo, hi) ↦ (lo·2^{k−n} + hi, hi·2^{k−n} − lo)`. -/
theorem slot2_exact (k n a : Nat) :
    (halvesCell (k := k) n a).2 = revHalves (k := k) n a := by
  simp only [halvesCell, halves, halvesMul, ringMul, revHalves]
  ring

/-- **The second coordinate in `ℕ` form**: the high digit scaled by `2^{k−n}`, less the low digit,
    the subtraction taken in `ZMod (2^k)`. -/
theorem revHalves_eq_sub (k n a : Nat) :
    revHalves (k := k) n a
      = denNat (k := k) (hiBits n a) * denNat (k := k) (2 ^ (k - n)) - denNat (k := k) (loBits n a) := rfl

/-- With a zero low digit there is no borrow, so the second slot is the `k`-bit word
    `⌊a/2^n⌋·2^{k−n}` itself. -/
theorem revHalves_of_loBits_zero {k n a : Nat} (h : loBits n a = 0) :
    revHalves (k := k) n a
      = denNat (k := k) (hiBits n a) * denNat (k := k) (2 ^ (k - n)) := by
  rw [revHalves_eq_sub, h]
  simp [denNat]

/-- The pairing loses nothing: the two digits recombine to the word. Re-proved here, so the
    "the forced pairing carries all `k` bits" claim is this file's own. -/
theorem halves_recover {k n a : Nat} :
    loBits n a + hiBits n a * denNat (k := k) (2 ^ n) = denNat (k := k) a := by
  have key : loBits n a + hiBits n a * denNat (k := k) (2 ^ n)
      = denNat (k := k) (a / 2 ^ n) * denNat (k := k) (2 ^ n) + denNat (k := k) (a % 2 ^ n) := by
    simp only [loBits, hiBits, denNat]
    ring
  rw [key]
  have hdiv := congrArg (fun t : Nat => (t : ZMod (2 ^ k))) (Nat.div_add_mod' a (2 ^ n))
  simpa only [denNat, Nat.cast_mul, Nat.cast_add] using hdiv

/-- **`rotrVal` is the extracted rotation's value**, for `a < 2^k` and `n < k`, so every statement
    above about `rotrVal` is a statement about SHA-256's `rotr` at an amount below the width.
    Re-derived from `BitVec.toNat_rotateRight`. -/
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

/-- A rotation value is a `k`-bit word. -/
theorem rotrVal_lt {k m a : Nat} (ha : a < 2 ^ k) (hm : m < k) :
    rotrVal (k := k) m a < 2 ^ k := by
  have hq : a / 2 ^ m < 2 ^ (k - m) := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos m), mul_comm (2 ^ (k - m)) (2 ^ m),
      ← Nat.pow_add]
    have hkey : m + (k - m) = k := by omega
    rw [hkey]
    exact ha
  have hstep : a / 2 ^ m + (a % 2 ^ m) * 2 ^ (k - m)
      < 2 ^ (k - m) + (a % 2 ^ m) * 2 ^ (k - m) := Nat.add_lt_add_right hq _
  have hone : a % 2 ^ m + 1 ≤ 2 ^ m := by
    omega
  calc rotrVal (k := k) m a
      = a / 2 ^ m + (a % 2 ^ m) * 2 ^ (k - m) := rfl
    _ < 2 ^ (k - m) + (a % 2 ^ m) * 2 ^ (k - m) := hstep
    _ = (a % 2 ^ m + 1) * 2 ^ (k - m) := by ring
    _ ≤ 2 ^ m * 2 ^ (k - m) :=
      Nat.mul_le_mul (n₁ := a % 2 ^ m + 1) (m₁ := 2 ^ (k - m)) (n₂ := 2 ^ m)
        (m₂ := 2 ^ (k - m)) hone (le_refl _)
    _ = 2 ^ k := by
        have hkey : m + (k - m) = k := by omega
        rw [← Nat.pow_add, hkey]

/-- **`shrVal` is the extracted shift's value**, for `a < 2^k` and `n < k`: `BitVec`'s `>>>` is the
    logical shift, so its value is `⌊a/2^n⌋`. This is the width-`k` statement about `SHR` that §2
    needs; it is proved here rather than assumed from AR-23, so **no result below is conditional on
    AR-23**. -/
theorem shrVal_eq_shrxBv {k : Nat} (a n : Nat) (ha : a < 2 ^ k) (hnk : n < k) :
    (shrxBv (k := k) a n).toNat = shrVal n a := by
  simp only [shrxBv, shrVal]
  rw [BitVec.toNat_ushiftRight, BitVec.toNat_ofNat, Nat.mod_eq_of_lt ha,
    Nat.shiftRight_eq_div_pow, Nat.mod_eq_of_lt hnk]

/-- `2 ≤ 2^j` for `j ≥ 1`. -/
theorem two_pow_ge_two (j : Nat) (h : 1 ≤ j) : (2 : Nat) ≤ 2 ^ j := by
  have hone : (1 : Nat) < 2 ^ j := Nat.one_lt_two_pow (by omega)
  omega

/-- `2^M < 2^n` for `M < n`. -/
theorem two_pow_lt_two_pow {n M : Nat} (h : M < n) : 2 ^ M < 2 ^ n := by
  have hkey : n = M + (n - M) := by omega
  rw [hkey, Nat.pow_add]
  calc 2 ^ M = 2 ^ M * 1 := by simp
    _ < 2 ^ M * 2 ^ (n - M) :=
      Nat.mul_lt_mul_of_pos_left (two_pow_ge_two (n - M) (by omega)) (by simp : (0 : Nat) < 2 ^ M)

/-- `2^M ≤ 2^n` for `M ≤ n`. -/
theorem two_pow_le_two_pow {n M : Nat} (h : M ≤ n) : 2 ^ M ≤ 2 ^ n := by
  rcases Nat.lt_or_eq_of_le h with h' | h'
  · exact Nat.le_of_lt (two_pow_lt_two_pow h')
  · rw [h']

/-- **The pairing degenerates at or above the width.** For `k ≤ n` the low digit is the whole
    `k`-bit word and the high digit is zero: the pairing *is* the one-slot pairing AR-12 refuted. -/
theorem halves_degenerate {k n a : Nat} (hn : k ≤ n) (ha : a < 2 ^ k) :
    halves (k := k) n a = (denNat (k := k) a, 0) := by
  have h2 : 2 ^ k ≤ 2 ^ n := two_pow_le_two_pow hn
  have hlo : loBits n a = a := by
    simp only [loBits]
    exact Nat.mod_eq_of_lt (lt_of_lt_of_le ha h2)
  have hhi : hiBits n a = 0 := by
    simp only [hiBits]
    exact Nat.div_eq_of_lt (lt_of_lt_of_le ha h2)
  simp only [halves, hlo, hhi, denNat, Nat.cast_zero, zero_mul]

/-- **And at an amount at or above the width the read is the word itself**, which is *not*
    SHA-256's rotation: at `k = 4` the pairing at amount `7` reads the word itself
    (`halves_width4_witness`). The FIPS regime therefore needs `k ≥ 26`, and the small widths
    cannot instantiate the FIPS amounts. -/
theorem halves_read_is_identity_above_width {k n a : Nat} (hn : k ≤ n) (ha : a < 2 ^ k) :
    rotrRead (k := k) n a = denNat (k := k) a := by
  have key : rotrRead (k := k) n a
      = denNat (k := k) a * denNat (k := k) (2 ^ (k - n)) := by
    simp only [rotrRead, halvesCell]
    rw [halves_degenerate hn ha]
    simp only [ringMul, denNat, halvesMul]
    ring
  have hone : denNat (k := k) (2 ^ (k - n)) = 1 := by
    have hkn : k - n = 0 := by omega
    simp only [hkn, pow_zero, denNat, Nat.cast_one]
  rw [key, hone, mul_one]

/-! ### §1b The multiplier's norm form: which FIPS amounts are rotations of the plane -/

/-- The forced multiplier's elliptic norm form: `N(2^{k−n}, −1) = 2^{2(k−n)} + 1`. -/
theorem halves_multiplier_norm_form {k n : Nat} :
    ringNormForm (-1 : ZMod (2 ^ k)) (halvesMul (k := k) n)
      = denNat (k := k) (2 ^ (k - n)) ^ 2 + 1 := by
  simp only [halvesMul, ringNormForm, denNat]
  ring

/-- **The multiplier is on the unit circle whenever the gain exponent reaches the width**,
    `k ≤ 2·(k−n)`: the read is then an isometry of the plane. -/
theorem halves_unitary_of {k n : Nat} (h : k ≤ 2 * (k - n)) :
    ringNormForm (-1 : ZMod (2 ^ k)) (halvesMul (k := k) n) = 1 := by
  rw [halves_multiplier_norm_form]
  have hsq : (denNat (k := k) (2 ^ (k - n)) : ZMod (2 ^ k)) ^ 2
      = denNat (k := k) (2 ^ (2 * (k - n))) := by
    simp only [denNat, Nat.cast_pow]
    exact (pow_mul' (2 : ZMod (2 ^ k)) 2 (k - n)).symm
  have hz : denNat (k := k) (2 ^ (2 * (k - n))) = 0 :=
    ZMod.natCast_eq_zero_iff _ _ |>.2
      ((Nat.pow_dvd_pow_iff_le_right (x := 2) (k := k) (l := 2 * (k - n)) (by norm_num)).2 h)
  rw [hsq, hz]
  ring

/-- **Off the circle otherwise.** AR-17's obstruction O-B with the equivalence removed: the read
    is still faithful (it is still `rotr n`, by `slot1_exact`), but the multiplication is not an
    isometry. -/
theorem halves_not_unitary_of {k n : Nat} (_hk : 1 ≤ k) (h : 2 * (k - n) < k) :
    ringNormForm (-1 : ZMod (2 ^ k)) (halvesMul (k := k) n) ≠ 1 := by
  intro hc
  have key : (denNat (k := k) (2 ^ (k - n)) : ZMod (2 ^ k)) ^ 2 + 1 = 1 := by
    simpa only [halves_multiplier_norm_form] using hc
  have hsq : (denNat (k := k) (2 ^ (k - n)) : ZMod (2 ^ k)) ^ 2
      = denNat (k := k) (2 ^ (2 * (k - n))) := by
    simp only [denNat, Nat.cast_pow]
    exact (pow_mul' (2 : ZMod (2 ^ k)) 2 (k - n)).symm
  have hz : denNat (k := k) (2 ^ (2 * (k - n))) = 0 := by
    rw [← hsq]
    linear_combination key
  have hd : 2 ^ k ∣ 2 ^ (2 * (k - n)) := ZMod.natCast_eq_zero_iff _ _ |>.1 hz
  have hd' : k ≤ 2 * (k - n) :=
    (Nat.pow_dvd_pow_iff_le_right (x := 2) (k := k) (l := 2 * (k - n)) (by norm_num)).1 hd
  omega

/-- **At the corpus width `k = 32` the FIPS rotation amounts split by the circle**: `2, 6, 7, 11,
    13` are unitary, `17, 18, 19, 22, 25` are faithful reads off it. The amounts `3` and `10` are
    shifts, not rotations, and are not rotation amounts at all. -/
theorem sha256_amounts_unitary :
    ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 2) = 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 6) = 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 7) = 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 11) = 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 13) = 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 17) ≠ 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 18) ≠ 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 19) ≠ 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 22) ≠ 1
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (halvesMul (k := 32) 25) ≠ 1 := by
  refine ⟨halves_unitary_of (k := 32) (n := 2) (by norm_num),
    halves_unitary_of (k := 32) (n := 6) (by norm_num),
    halves_unitary_of (k := 32) (n := 7) (by norm_num),
    halves_unitary_of (k := 32) (n := 11) (by norm_num),
    halves_unitary_of (k := 32) (n := 13) (by norm_num),
    halves_not_unitary_of (k := 32) (n := 17) (by norm_num) (by norm_num),
    halves_not_unitary_of (k := 32) (n := 18) (by norm_num) (by norm_num),
    halves_not_unitary_of (k := 32) (n := 19) (by norm_num) (by norm_num),
    halves_not_unitary_of (k := 32) (n := 22) (by norm_num) (by norm_num),
    halves_not_unitary_of (k := 32) (n := 25) (by norm_num) (by norm_num)⟩

/-! ## §2 The pairing is amount-specific: the span of the two digit coordinates -/

/-- `2^j % 2 = 0` for `j ≥ 1`. -/
theorem two_pow_mod_two (j : Nat) (h : 1 ≤ j) : 2 ^ j % 2 = 0 := by
  obtain ⟨m, rfl⟩ : ∃ m, j = m + 1 := ⟨j - 1, by omega⟩
  have hp : 2 ^ (m + 1) = 2 ^ m * 2 := Nat.pow_succ ..
  rw [hp, Nat.mul_mod, Nat.mod_self, Nat.mul_zero, Nat.zero_mod]

/-- `2^j % 2 = if j = 0 then 1 else 0`. -/
theorem two_pow_mod_two' (j : Nat) : 2 ^ j % 2 = if j = 0 then 1 else 0 := by
  by_cases hj : j = 0
  · rw [hj, pow_zero, Nat.one_mod, if_pos rfl]
  · rw [two_pow_mod_two j (by omega)]
    simp [hj]

/-- The low bit of `⊕` is the sum of the low bits. -/
theorem xor_mod_two (a b : Nat) : (a ^^^ b) % 2 = a % 2 ^^^ b % 2 := by
  simpa using (Nat.xor_mod_two_pow (a := a) (b := b) (n := 1))

/-- The low bit of a three-fold `⊕`, given the three low bits. -/
theorem xor3_mod_two (a b c α β γ : Nat) (_hα : α < 2) (_hβ : β < 2) (_hγ : γ < 2)
    (ha : a % 2 = α) (hb : b % 2 = β) (hc : c % 2 = γ) :
    (a ^^^ (b ^^^ c)) % 2 = α ^^^ (β ^^^ γ) := by
  rw [xor_mod_two, xor_mod_two, ha, hb, hc]

/-- The low digit of a single-bit word: zero below the split, the bit above it. -/
theorem loBits_pow2 (n M : Nat) : loBits n (2 ^ M) = if n ≤ M then 0 else 2 ^ M := by
  by_cases h : n ≤ M
  · rw [if_pos h, loBits,
      Nat.mod_eq_zero_of_dvd ((Nat.pow_dvd_pow_iff_le_right (x := 2) (k := n) (l := M)
        (by norm_num)).2 h)]
  · rw [if_neg h, loBits, Nat.mod_eq_of_lt (two_pow_lt_two_pow (by omega))]

/-- The high digit of a single-bit word: the bit above the split, zero above it. -/
theorem hiBits_pow2 (n M : Nat) : hiBits n (2 ^ M) = if n ≤ M then 2 ^ (M - n) else 0 := by
  by_cases h : n ≤ M
  · rw [if_pos h, hiBits,
      Nat.div_eq_iff_eq_mul_left (Nat.two_pow_pos n)
        ((Nat.pow_dvd_pow_iff_le_right (x := 2) (k := n) (l := M) (by norm_num)).2 h),
      ← Nat.pow_add]
    congr 1
    omega
  · rw [if_neg h, hiBits, Nat.div_eq_of_lt (two_pow_lt_two_pow (by omega))]

/-- **Both digits of `2^M` are even at every split other than `M` itself.** This is the
    ground of every refutation in §2 and §3: a word whose cell is *both even* cannot be separated
    from zero by a linear functional of the cell, while a rotation or shift at its own amount
    sends it to the odd word `1`. -/
theorem halves_pow2_even (n M : Nat) (hM : 1 ≤ M) (hne : n ≠ M) :
    loBits n (2 ^ M) % 2 = 0 ∧ hiBits n (2 ^ M) % 2 = 0 := by
  constructor
  · rw [loBits_pow2]
    split
    · exact Nat.zero_mod 2
    · exact two_pow_mod_two M hM
  · rw [hiBits_pow2]
    by_cases h : n ≤ M
    · rw [if_pos h]
      exact two_pow_mod_two (M - n) (by omega)
    · rw [if_neg h]

/-- `rotrVal` in the high/low digit form; definitional, recorded so that §2 rewrites do not
    re-open the `ℕ` definition. -/
theorem rotrVal_eq_hi_lo (k m a : Nat) : rotrVal (k := k) m a = hiBits m a + loBits m a * 2 ^ (k - m) := rfl

/-- **The low bit of a rotation at a single-bit word**: bit `M` of the word moves to bit `0`
    only when the amount *is* `M`; otherwise the rotation's value is even. -/
theorem rotrVal_pow2_mod_two (k M m : Nat) (_hk : 1 ≤ k) (hm : m < k) :
    rotrVal (k := k) m (2 ^ M) % 2 = if M = m then 1 else 0 := by
  rcases Nat.lt_or_ge M m with hMm | hMm
  · rw [rotrVal_eq_hi_lo, hiBits_pow2, loBits_pow2, if_neg (by omega : ¬ m ≤ M),
      if_neg (by omega : ¬ m ≤ M), Nat.zero_add, ← Nat.pow_add]
    rw [two_pow_mod_two _ (by omega)]
    simp [Nat.ne_of_lt hMm]
  · rw [rotrVal_eq_hi_lo, hiBits_pow2, loBits_pow2, if_pos hMm, if_pos hMm,
      Nat.zero_mul, Nat.add_zero, two_pow_mod_two']
    by_cases h0 : M - m = 0
    · have hmm : M = m := by omega
      simp [h0, hmm]
    · have hmm : M ≠ m := by omega
      simp [h0, hmm]

/-- **The low bit of a shift at a single-bit word**: `1` exactly at the shift's own amount. -/
theorem shrVal_pow2_mod_two (M r : Nat) : shrVal r (2 ^ M) % 2 = if M = r then 1 else 0 := by
  show hiBits r (2 ^ M) % 2 = if M = r then 1 else 0
  rw [hiBits_pow2]
  by_cases h : r ≤ M
  · rw [if_pos h, two_pow_mod_two']
    by_cases h0 : M - r = 0
    · have hmm : M = r := by omega
      simp [h0, hmm]
    · have hmm : M ≠ r := by omega
      simp [h0, hmm]
  · rw [if_neg h]
    have hne : M ≠ r := by rintro rfl; omega
    simp [hne]

/-- **A rotation at its own amount sends the single-bit word `2^m` to `1`.** -/
theorem rotrVal_pow2_self (k m : Nat) (_hm : m < k) : rotrVal (k := k) m (2 ^ m) = 1 := by
  rw [rotrVal_eq_hi_lo, hiBits_pow2, loBits_pow2, if_pos (by omega), if_pos (by omega),
    Nat.zero_mul, Nat.add_zero]
  simp

/-- **A shift at its own amount sends the single-bit word `2^r` to `1`.** -/
theorem shrVal_pow2_self (_M r : Nat) : shrVal r (2 ^ r) = 1 := by
  show hiBits r (2 ^ r) = 1
  rw [hiBits_pow2, if_pos (by omega)]
  simp

/-- **The parity obstruction at the ring level.** An odd target cannot be produced from two even
    digit coordinates by any `ZMod`-linear functional: the residue mod `2` of a sum of
    `ZMod`-multiples of even numbers is `0`, and that of the odd target is `1`. No face parameter
    appears, so no σ escapes it. -/
theorem zmod_ne_of_parity {k : Nat} (hk : 1 ≤ k) (L H F : Nat)
    (hL : L % 2 = 0) (hH : H % 2 = 0) (hF : F % 2 = 1) (c₁ c₂ : ZMod (2 ^ k)) :
    c₁ * denNat (k := k) L + c₂ * denNat (k := k) H ≠ denNat (k := k) F := by
  have hdv : (2 : Nat) ∣ 2 ^ k :=
    (Nat.pow_dvd_pow_iff_le_right (x := 2) (k := 1) (l := k) (by norm_num)).2 hk
  have hmod2 (x : Nat) : (x % 2 ^ k) % 2 = x % 2 := by
    calc (x % 2 ^ k) % 2 = (x % 2) % 2 ^ k := (Nat.mod_mod_eq_mod_mod_of_dvd hdv).symm
      _ = x % 2 := Nat.mod_mod_of_dvd' hdv
  have hden (x : Nat) : (denNat (k := k) x).val = x % 2 ^ k := ZMod.val_natCast _ _
  intro hcon
  have hv1 : (c₁ * denNat (k := k) L).val % 2 = 0 := by
    rw [ZMod.val_mul, hden, hmod2, Nat.mul_mod, hmod2, hL, Nat.mul_zero, Nat.zero_mod]
  have hv2 : (c₂ * denNat (k := k) H).val % 2 = 0 := by
    rw [ZMod.val_mul, hden, hmod2, Nat.mul_mod, hmod2, hH, Nat.mul_zero, Nat.zero_mod]
  have hsum : (c₁ * denNat (k := k) L + c₂ * denNat (k := k) H).val % 2 = 0 := by
    rw [ZMod.val_add, hmod2, Nat.add_mod, hv1, hv2]
  have hval := congrArg ZMod.val hcon
  rw [hval, hden, hmod2] at hsum
  omega

/-- **THE CLASS THE FORCED PAIRING CAN READ.** `Span2 k n g`: `g` is the first slot of a
    `ZMod`-linear functional of the two digit coordinates of the `n`-split pairing — that is,
    `c₁ · (a mod 2^n) + c₂ · ⌊a/2^n⌋` for some `c₁, c₂ ∈ ZMod (2^k)`. Every read AR-17 and AR-12
    considered (first slot of a `ringMul`, a `ringUpper` of it, `ringNormForm`) is in this class,
    because each is `ZMod`-linear in the cell's coordinates. -/
def Span2 {k : Nat} (n : Nat) (g : Nat → Nat) : Prop :=
  ∃ c₁ c₂ : ZMod (2 ^ k), ∀ a : Nat, a < 2 ^ k →
    c₁ * denNat (k := k) (loBits n a) + c₂ * denNat (k := k) (hiBits n a) = denNat (k := k) (g a)

/-- **THE OBSTRUCTION, IN THE HALVES COORDINATES. An odd value at a word whose two digit
    coordinates are both even is not a `ZMod`-linear functional of the cell**, at any width and
    any split. This is the statement AR-12 could not make, because its XOR results
    (`zxor_not_ringAdd`, `zxor_const_not_affine`, `zxor_rotr_not_affine`) were about the *word*
    ring `ZMod (2^k)` and the one-slot family, and this one is about the *digit coordinates*. -/
theorem not_span_of_evenCell_oddVal {k n w : Nat} (hk : 1 ≤ k) (hw : w < 2 ^ k)
    (hcL : loBits n w % 2 = 0) (hcH : hiBits n w % 2 = 0) (g : Nat → Nat) (hg : g w % 2 = 1) :
    ¬ Span2 (k := k) n g := by
  rintro ⟨c₁, c₂, h⟩
  exact zmod_ne_of_parity hk (loBits n w) (hiBits n w) (g w) hcL hcH hg c₁ c₂ (h w hw)

/-- **THE PAIRING SERVES EXACTLY ITS OWN AMOUNT, for rotations.** A rotation by `m` is a
    `ZMod`-linear functional of the `n`-split cell exactly when `n = m`, at every width. The
    refutation's witness is the single-bit word `2^m`, where the rotation reads `1` and both digit
    coordinates are even for every `n ≠ m`. -/
theorem span_rotr_iff {k m n : Nat} (hk : 1 ≤ k) (hm : 0 < m) (hmk : m < k) (hn : 1 ≤ n)
    (hnk : n < k) : Span2 (k := k) n (rotrVal (k := k) m) ↔ m = n := by
  constructor
  · intro hsp
    by_contra hne
    have hcell : loBits n (2 ^ m) % 2 = 0 ∧ hiBits n (2 ^ m) % 2 = 0 :=
      halves_pow2_even (n := n) (M := m) (by omega) (fun h => hne h.symm)
    have hodd : rotrVal (k := k) m (2 ^ m) % 2 = 1 := by
      rw [rotrVal_pow2_self k m hmk, Nat.one_mod]
    exact not_span_of_evenCell_oddVal (k := k) (n := n) (w := 2 ^ m) hk
      (two_pow_lt_two_pow (n := k) (M := m) hmk) hcell.1 hcell.2 (rotrVal (k := k) m) hodd hsp
  · intro h
    subst h
    refine ⟨denNat (k := k) (2 ^ (k - m)), denNat 1, fun a ha => ?_⟩
    rw [denNat_rotrVal]
    simp only [denNat, Nat.cast_one, one_mul]
    ring

/-- **AND FOR THE SHIFT.** The `SHR` term is a `ZMod`-linear functional of the `n`-split cell
    exactly at `n = r`, where it is literally the second coordinate
    (`shrRead_is_second_coordinate`). No conditionality on AR-23 is required. -/
theorem span_shr_iff {k r n : Nat} (hk : 1 ≤ k) (hr : 0 < r) (hrk : r < k) (hn : 1 ≤ n)
    (hnk : n < k) : Span2 (k := k) n (shrVal r) ↔ r = n := by
  constructor
  · intro hsp
    by_contra hne
    have hcell : loBits n (2 ^ r) % 2 = 0 ∧ hiBits n (2 ^ r) % 2 = 0 :=
      halves_pow2_even (n := n) (M := r) (by omega) (fun h => hne h.symm)
    have hzero : shrVal r (2 ^ r) = 1 := by
      show (2 ^ r : Nat) / 2 ^ r = 1
      exact Nat.div_self (Nat.two_pow_pos r)
    have hodd : shrVal r (2 ^ r) % 2 = 1 := by rw [hzero, Nat.one_mod]
    exact not_span_of_evenCell_oddVal (k := k) (n := n) (w := 2 ^ r) hk
      (two_pow_lt_two_pow (n := k) (M := r) hrk) hcell.1 hcell.2 (shrVal r) hodd hsp
  · intro h
    subst h
    refine ⟨0, 1, fun a ha => ?_⟩
    simp only [shrVal, hiBits]
    ring

/-- **The `SHR` term is the second coordinate of its own split**, so it costs a projection and no
    ring operation at all. -/
theorem shrRead_is_second_coordinate (k r a : Nat) :
    denNat (k := k) (shrVal r a) = (halves (k := k) r a).2 := by
  simp [halves, hiBits, shrVal]

/-! ### §2b Ground tables at `k = 4` and `k = 2`

The refutations of §2 are proved for every width and every pair of amounts, so their `k = 4` and
`k = 2` instances are *instances* of those theorems, listed in §2c rather than re-decided. What is
decided at small width here is the per-term read and the second-slot content, over every word. -/

/-- **The width-4 table of the per-term read**: every amount `0 … 4` and every four-bit word. -/
theorem slot1_table_width4 : ∀ (n : Fin 5) (a : Fin 16),
    rotrRead (k := 4) n.val a.val = denNat (k := 4) (rotrVal (k := 4) n.val a.val) := by
  decide

/-- **The width-4 table of the second-coordinate content.** -/
theorem slot2_table_width4 : ∀ (n : Fin 5) (a : Fin 16),
    (halvesCell (k := 4) n.val a.val).2 = revHalves (k := 4) n.val a.val := by
  decide

/-- **The width-2 table of the per-term read.** -/
theorem slot1_table_width2 : ∀ (n : Fin 3) (a : Fin 4),
    rotrRead (k := 2) n.val a.val = denNat (k := 2) (rotrVal (k := 2) n.val a.val) := by
  decide

/-- **At `k = 4` the amount `7` is above the width, and the pairing reads the word instead of
    rotating it**: on the word `1` the pairing reads `1` and the extracted body reads `2`. The
    ground check of `halves_read_is_identity_above_width`. -/
theorem halves_width4_witness :
    rotrRead (k := 4) 7 1 = (1 : ZMod 16) ∧ (rotrBv (k := 4) 1 7).toNat = 2 := by
  decide

/-! ### §2c The `k = 4` and `k = 2` instances of §2's refutations -/

/-- **Width 4: the rotation by `3` is a linear functional of the `3`-split cell and of no other
    split's cell** — `span_rotr_iff` at `k = 4`, discharged by instantiation. -/
theorem span_rotr_width4_own : Span2 (k := 4) 3 (rotrVal (k := 4) 3) :=
  (span_rotr_iff (k := 4) (m := 3) (n := 3) (by omega) (by omega) (by omega) (by omega)
    (by omega)).2 rfl

/-- **Width 4: it is *not* a linear functional of the `1`-split cell.** -/
theorem span_rotr_width4_not : ¬ Span2 (k := 4) 1 (rotrVal (k := 4) 3) := by
  intro hsp
  have hne : (3 : Nat) = 1 :=
    (span_rotr_iff (k := 4) (m := 3) (n := 1) (by omega) (by omega) (by omega) (by omega)
      (by omega)).mp hsp
  omega

/-- **Width 4: the shift by `3` is a linear functional of the `3`-split cell only.** -/
theorem span_shr_width4_own : Span2 (k := 4) 3 (shrVal 3) :=
  (span_shr_iff (k := 4) (r := 3) (n := 3) (by omega) (by omega) (by omega) (by omega)
    (by omega)).2 rfl

/-- **Width 4: the shift by `3` is not a linear functional of the `2`-split cell.** -/
theorem span_shr_width4_not : ¬ Span2 (k := 4) 2 (shrVal 3) := by
  intro hsp
  have hne : (3 : Nat) = 2 :=
    (span_shr_iff (k := 4) (r := 3) (n := 2) (by omega) (by omega) (by omega) (by omega)
      (by omega)).mp hsp
  omega

/-! ## §3 The XOR layer, in the halves coordinates -/

/-- `Σ₀(x) = ROTR^2 x ⊕ ROTR^13 x ⊕ ROTR^22 x`, at width `k`: the corpus's `Sigma0` with `32`
    replaced by a parameter, in `ℕ` values. -/
def bigSigma0Val (k a : Nat) : Nat :=
  rotrVal (k := k) 2 a ^^^ (rotrVal (k := k) 13 a ^^^ rotrVal (k := k) 22 a)

/-- `Σ₁(x) = ROTR^6 x ⊕ ROTR^11 x ⊕ ROTR^25 x`. -/
def bigSigma1Val (k a : Nat) : Nat :=
  rotrVal (k := k) 6 a ^^^ (rotrVal (k := k) 11 a ^^^ rotrVal (k := k) 25 a)

/-- `σ₀(x) = ROTR^7 x ⊕ ROTR^18 x ⊕ SHR^3 x`: the first two terms are rotations, the third a
    shift. -/
def smallSigma0Val (k a : Nat) : Nat :=
  rotrVal (k := k) 7 a ^^^ (rotrVal (k := k) 18 a ^^^ shrVal 3 a)

/-- `σ₁(x) = ROTR^17 x ⊕ ROTR^19 x ⊕ SHR^10 x`. -/
def smallSigma1Val (k a : Nat) : Nat :=
  rotrVal (k := k) 17 a ^^^ (rotrVal (k := k) 19 a ^^^ shrVal 10 a)

/-- The low bit of a three-fold `⊕` of rotations at the single-bit word `2^p`, with the other two
    amounts distinct from `p`: the value is odd. -/
theorem xor3_rot_odd {k p q r : Nat} (x : Nat) (h1 : rotrVal (k := k) p x % 2 = 1)
    (h2 : rotrVal (k := k) q x % 2 = 0) (h3 : rotrVal (k := k) r x % 2 = 0) :
    (rotrVal (k := k) p x ^^^ (rotrVal (k := k) q x ^^^ rotrVal (k := k) r x)) % 2 = 1 := by
  have hx := xor3_mod_two (rotrVal (k := k) p x) (rotrVal (k := k) q x) (rotrVal (k := k) r x)
    1 0 0 (by decide) (by decide) (by decide) h1 h2 h3
  simpa using hx

/-- The same for two rotations and a shift. -/
theorem xor3_rot_shr_odd {k p q r : Nat} (x : Nat) (h1 : rotrVal (k := k) p x % 2 = 1)
    (h2 : rotrVal (k := k) q x % 2 = 0) (h3 : shrVal r x % 2 = 0) :
    (rotrVal (k := k) p x ^^^ (rotrVal (k := k) q x ^^^ shrVal r x)) % 2 = 1 := by
  have hx := xor3_mod_two (rotrVal (k := k) p x) (rotrVal (k := k) q x) (shrVal r x)
    1 0 0 (by decide) (by decide) (by decide) h1 h2 h3
  simpa using hx

/-- The low bit of a three-fold `⊕` of rotations at a word, given the three low bits. -/
theorem xor3_rot_odd' {k p q r : Nat} (x : Nat) (α β γ : Nat) (_hα : α < 2) (_hβ : β < 2)
    (_hγ : γ < 2) (h1 : rotrVal (k := k) p x % 2 = α) (h2 : rotrVal (k := k) q x % 2 = β)
    (h3 : rotrVal (k := k) r x % 2 = γ) (hxor : α ^^^ (β ^^^ γ) = 1) :
    (rotrVal (k := k) p x ^^^ (rotrVal (k := k) q x ^^^ rotrVal (k := k) r x)) % 2 = 1 := by
  have hx := xor3_mod_two (rotrVal (k := k) p x) (rotrVal (k := k) q x) (rotrVal (k := k) r x)
    α β γ (by omega) (by omega) (by omega) h1 h2 h3
  simpa [hxor] using hx

/-- The same for two rotations and a shift. -/
theorem xor3_rot_shr_odd' {k p q r : Nat} (x : Nat) (α β γ : Nat) (_hα : α < 2) (_hβ : β < 2)
    (_hγ : γ < 2) (h1 : rotrVal (k := k) p x % 2 = α) (h2 : rotrVal (k := k) q x % 2 = β)
    (h3 : shrVal r x % 2 = γ) (hxor : α ^^^ (β ^^^ γ) = 1) :
    (rotrVal (k := k) p x ^^^ (rotrVal (k := k) q x ^^^ shrVal r x)) % 2 = 1 := by
  have hx := xor3_mod_two (rotrVal (k := k) p x) (rotrVal (k := k) q x) (shrVal r x)
    α β γ (by omega) (by omega) (by omega) h1 h2 h3
  simpa [hxor] using hx

/-- **THE XOR-LAYER OBSTRUCTION, THREE ROTATIONS, WITNESS CHOSEN.** Distinct amounts `p, q, r` below
    the width, and a witness exponent `w ∈ {p, q}` distinct from the split `n`: the `⊕` of the
    three is not a `ZMod`-linear functional of the `n`-split cell, at every width. The witness is
    the single-bit word `2^w`: exactly one of the three terms reads `1` there, and both its digit
    coordinates are even. -/
theorem span_xor3_not_of_distinct_at {k n p q r w : Nat} (hk : 1 ≤ k) (hpk : p < k)
    (hqk : q < k) (hrk : r < k) (hw1 : 1 ≤ w) (hwk : w < k) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r)
    (hnw : n ≠ w) (hwq : w = q ∨ w = p) :
    ¬ Span2 (k := k) n (fun a => rotrVal (k := k) p a ^^^ (rotrVal (k := k) q a ^^^ rotrVal (k := k) r a)) := by
  have hcell : loBits n (2 ^ w) % 2 = 0 ∧ hiBits n (2 ^ w) % 2 = 0 :=
    halves_pow2_even (n := n) (M := w) hw1 (fun h => hnw h)
  have h1 : rotrVal (k := k) p (2 ^ w) % 2 = if w = p then 1 else 0 := rotrVal_pow2_mod_two k w p hk hpk
  have h2 : rotrVal (k := k) q (2 ^ w) % 2 = if w = q then 1 else 0 := rotrVal_pow2_mod_two k w q hk hqk
  have h3 : rotrVal (k := k) r (2 ^ w) % 2 = if w = r then 1 else 0 := rotrVal_pow2_mod_two k w r hk hrk
  refine not_span_of_evenCell_oddVal (k := k) (n := n) (w := 2 ^ w) hk
    (two_pow_lt_two_pow (n := k) (M := w) hwk) hcell.1 hcell.2
    (fun a => rotrVal (k := k) p a ^^^ (rotrVal (k := k) q a ^^^ rotrVal (k := k) r a)) ?_
  rcases hwq with hwq | hwp
  · rw [hwq] at h1 h2 h3
    rw [if_neg (Ne.symm hpq)] at h1
    rw [if_pos rfl] at h2
    rw [if_neg hqr] at h3
    simpa [hwq] using
      (xor3_rot_odd' (2 ^ q) 0 1 0 (by decide) (by decide) (by decide) h1 h2 h3 rfl)
  · rw [hwp] at h1 h2 h3
    rw [if_pos rfl] at h1
    rw [if_neg hpq] at h2
    rw [if_neg hpr] at h3
    simpa [hwp] using
      (xor3_rot_odd' (2 ^ p) 1 0 0 (by decide) (by decide) (by decide) h1 h2 h3 rfl)

/-- **THE XOR-LAYER OBSTRUCTION, TWO ROTATIONS AND THE SHIFT, WITNESS CHOSEN.** The `SHR` term is
    included, so this covers the schedule's `σ₀` and `σ₁` without any hypothesis on the shift's
    algebra. -/
theorem span_xor2rot_shr_not_of_distinct_at {k n p q r w : Nat} (hk : 1 ≤ k) (hpk : p < k)
    (hqk : q < k) (hw1 : 1 ≤ w) (hwk : w < k) (hpq : p ≠ q) (hqr : q ≠ r) (hwr : w ≠ r)
    (hnw : n ≠ w) (hwq : w = q ∨ w = p) :
    ¬ Span2 (k := k) n (fun a => rotrVal (k := k) p a ^^^ (rotrVal (k := k) q a ^^^ shrVal r a)) := by
  have hcell : loBits n (2 ^ w) % 2 = 0 ∧ hiBits n (2 ^ w) % 2 = 0 :=
    halves_pow2_even (n := n) (M := w) hw1 (fun h => hnw h)
  have h1 : rotrVal (k := k) p (2 ^ w) % 2 = if w = p then 1 else 0 := rotrVal_pow2_mod_two k w p hk hpk
  have h2 : rotrVal (k := k) q (2 ^ w) % 2 = if w = q then 1 else 0 := rotrVal_pow2_mod_two k w q hk hqk
  have h3 : shrVal r (2 ^ w) % 2 = if w = r then 1 else 0 := shrVal_pow2_mod_two w r
  refine not_span_of_evenCell_oddVal (k := k) (n := n) (w := 2 ^ w) hk
    (two_pow_lt_two_pow (n := k) (M := w) hwk) hcell.1 hcell.2
    (fun a => rotrVal (k := k) p a ^^^ (rotrVal (k := k) q a ^^^ shrVal r a)) ?_
  rcases hwq with hwq | hwp
  · rw [hwq] at h1 h2 h3 hwr
    rw [if_neg (Ne.symm hpq)] at h1
    rw [if_pos rfl] at h2
    rw [if_neg hqr] at h3
    simpa [hwq] using
      (xor3_rot_shr_odd' (2 ^ q) 0 1 0 (by decide) (by decide) (by decide) h1 h2 h3 rfl)
  · rw [hwp] at h1 h2 h3 hwr
    rw [if_pos rfl] at h1
    rw [if_neg hpq] at h2
    rw [if_neg hwr] at h3
    simpa [hwp] using
      (xor3_rot_shr_odd' (2 ^ p) 1 0 0 (by decide) (by decide) (by decide) h1 h2 h3 rfl)

/-- **THE XOR-LAYER OBSTRUCTION, THREE ROTATIONS. Distinct amounts `p, q, r` below the width: the
    `⊕` of the three is not a `ZMod`-linear functional of the `n`-split cell, for every `n ≠ p`,
    at every width.** The witness is the single-bit word `2^p`: the `⊕` reads `1` there and both
    its digit coordinates are even. -/
theorem span_xor3_not_of_distinct {k n p q r : Nat} (hk : 1 ≤ k) (hp : 0 < p) (hpk : p < k)
    (hqk : q < k) (hrk : r < k) (hpq : p ≠ q) (hpr : p ≠ r) (hnp : n ≠ p) :
    ¬ Span2 (k := k) n (fun a => rotrVal (k := k) p a ^^^ (rotrVal (k := k) q a ^^^ rotrVal (k := k) r a)) := by
  have hcell : loBits n (2 ^ p) % 2 = 0 ∧ hiBits n (2 ^ p) % 2 = 0 :=
    halves_pow2_even (n := n) (M := p) (by omega) (fun h => hnp h)
  have h1 : rotrVal (k := k) p (2 ^ p) % 2 = 1 := by
    rw [rotrVal_pow2_self k p hpk, Nat.one_mod]
  have h2 : rotrVal (k := k) q (2 ^ p) % 2 = 0 := by
    rw [rotrVal_pow2_mod_two k p q hk hqk, if_neg hpq]
  have h3 : rotrVal (k := k) r (2 ^ p) % 2 = 0 := by
    rw [rotrVal_pow2_mod_two k p r hk hrk, if_neg hpr]
  refine not_span_of_evenCell_oddVal (k := k) (n := n) (w := 2 ^ p) hk
    (two_pow_lt_two_pow (n := k) (M := p) hpk) hcell.1 hcell.2
    (fun a => rotrVal (k := k) p a ^^^ (rotrVal (k := k) q a ^^^ rotrVal (k := k) r a))
    (xor3_rot_odd _ h1 h2 h3)

/-- **THE XOR-LAYER OBSTRUCTION, TWO ROTATIONS AND THE SHIFT.** The `SHR` term is included, so this
    covers the schedule's `σ₀` and `σ₁` without any hypothesis on the shift's algebra. -/
theorem span_xor2rot_shr_not_of_distinct {k n p q r : Nat} (hk : 1 ≤ k) (hp : 0 < p) (hpk : p < k)
    (hqk : q < k) (hpq : p ≠ q) (hpr : p ≠ r) (hnp : n ≠ p) :
    ¬ Span2 (k := k) n (fun a => rotrVal (k := k) p a ^^^ (rotrVal (k := k) q a ^^^ shrVal r a)) := by
  have hcell : loBits n (2 ^ p) % 2 = 0 ∧ hiBits n (2 ^ p) % 2 = 0 :=
    halves_pow2_even (n := n) (M := p) (by omega) (fun h => hnp h)
  have h1 : rotrVal (k := k) p (2 ^ p) % 2 = 1 := by
    rw [rotrVal_pow2_self k p hpk, Nat.one_mod]
  have h2 : rotrVal (k := k) q (2 ^ p) % 2 = 0 := by
    rw [rotrVal_pow2_mod_two k p q hk hqk, if_neg hpq]
  have h3 : shrVal r (2 ^ p) % 2 = 0 := by
    rw [shrVal_pow2_mod_two, if_neg hpr]
  refine not_span_of_evenCell_oddVal (k := k) (n := n) (w := 2 ^ p) hk
    (two_pow_lt_two_pow (n := k) (M := p) hpk) hcell.1 hcell.2
    (fun a => rotrVal (k := k) p a ^^^ (rotrVal (k := k) q a ^^^ shrVal r a))
    (xor3_rot_shr_odd _ h1 h2 h3)

/-- A shift value is a `k`-bit word, for every amount. -/
theorem shrVal_lt {k r a : Nat} (ha : a < 2 ^ k) : shrVal r a < 2 ^ k := by
  simp only [shrVal]
  exact lt_of_le_of_lt (Nat.div_le_self _ _) ha

/-! ### §3a The composite as three halves-pairing reads plus the named residual `zxor3` -/

/-- The read's representative: for a `k`-bit word and an amount below the width, the residue
    `rotrRead n a` *is* the `k`-bit value `rotrVal n a`. -/
theorem rotrRead_val {k n a : Nat} (ha : a < 2 ^ k) (hn : n < k) :
    (rotrRead (k := k) n a).val = rotrVal (k := k) n a := by
  rw [rotrRead_eq_denNat, denNat, ZMod.val_natCast_of_lt (rotrVal_lt ha hn)]


/-- **The composite `Σ₀` is three halves-pairing reads and the `zxor3` layer — nothing else.** Each
    of the three terms is `rotrRead` at its own amount, i.e. the first slot of the forced product;
    the residual is exactly the `⊕`. There is no per-term residual. -/
theorem bigSigma0_reads {k a : Nat} (hk : 26 ≤ k) (ha : a < 2 ^ k) :
    denNat (k := k) (bigSigma0Val k a)
      = zxor3 (rotrRead (k := k) 2 a) (rotrRead (k := k) 13 a) (rotrRead (k := k) 22 a) := by
  simp only [bigSigma0Val, zxor3]
  rw [rotrRead_val ha (by omega), rotrRead_val ha (by omega), rotrRead_val ha (by omega)]
  simp only [denNat]
  rw [Nat.xor_assoc]

/-- **The composite `Σ₁`.** -/
theorem bigSigma1_reads {k a : Nat} (hk : 26 ≤ k) (ha : a < 2 ^ k) :
    denNat (k := k) (bigSigma1Val k a)
      = zxor3 (rotrRead (k := k) 6 a) (rotrRead (k := k) 11 a) (rotrRead (k := k) 25 a) := by
  simp only [bigSigma1Val, zxor3]
  rw [rotrRead_val ha (by omega), rotrRead_val ha (by omega), rotrRead_val ha (by omega)]
  simp only [denNat]
  rw [Nat.xor_assoc]

/-- **The composite `σ₀`: two halves-pairing reads, the `SHR` term as a projection of its own
    3-split cell, and the `zxor3` layer.** -/
theorem smallSigma0_reads {k a : Nat} (hk : 26 ≤ k) (ha : a < 2 ^ k) :
    denNat (k := k) (smallSigma0Val k a)
      = zxor3 (rotrRead (k := k) 7 a) (rotrRead (k := k) 18 a) (denNat (k := k) (shrVal 3 a)) := by
  simp only [smallSigma0Val, zxor3]
  rw [rotrRead_val ha (by omega), rotrRead_val ha (by omega)]
  simp only [denNat]
  rw [ZMod.val_natCast_of_lt (shrVal_lt ha), Nat.xor_assoc]

/-- **The composite `σ₁`.** -/
theorem smallSigma1_reads {k a : Nat} (hk : 26 ≤ k) (ha : a < 2 ^ k) :
    denNat (k := k) (smallSigma1Val k a)
      = zxor3 (rotrRead (k := k) 17 a) (rotrRead (k := k) 19 a) (denNat (k := k) (shrVal 10 a)) := by
  simp only [smallSigma1Val, zxor3]
  rw [rotrRead_val ha (by omega), rotrRead_val ha (by omega)]
  simp only [denNat]
  rw [ZMod.val_natCast_of_lt (shrVal_lt ha), Nat.xor_assoc]

/-! ### §3b The obstruction: none of the four composites is a linear functional of the cell -/

/-- **`Σ₀` is not a `ZMod`-linear functional of the `n`-split cell, for every `k ≥ 26` and every
    split `1 ≤ n < k`.** Witness: `a = 2^2` for `n ≠ 2`, `a = 2^13` for `n = 2`. -/
theorem bigSigma0_not_span {k n : Nat} (hk : 26 ≤ k) (_hn : 1 ≤ n) (_hnk : n < k) :
    ¬ Span2 (k := k) n (bigSigma0Val k) := by
  by_cases h : n = 2
  · refine span_xor3_not_of_distinct_at (k := k) (n := n) (p := 2) (q := 13) (r := 22) (w := 13)
      (hwq := Or.inl rfl) (hnw := by omega)
      (by omega) (by omega) (by omega) (by omega) (by decide) (by omega) (by decide) (by decide)
      (by decide)
  · refine span_xor3_not_of_distinct_at (k := k) (n := n) (p := 2) (q := 13) (r := 22) (w := 2)
      (hwq := Or.inr rfl) (hnw := h)
      (by omega) (by omega) (by omega) (by omega) (by decide) (by omega) (by decide) (by decide)
      (by decide)

/-- **`Σ₁`**, with witnesses `2^6` (for `n ≠ 6`) and `2^11` (for `n = 6`). -/
theorem bigSigma1_not_span {k n : Nat} (hk : 26 ≤ k) (_hn : 1 ≤ n) (_hnk : n < k) :
    ¬ Span2 (k := k) n (bigSigma1Val k) := by
  by_cases h : n = 6
  · refine span_xor3_not_of_distinct_at (k := k) (n := n) (p := 6) (q := 11) (r := 25) (w := 11)
      (hwq := Or.inl rfl) (hnw := by omega)
      (by omega) (by omega) (by omega) (by omega) (by decide) (by omega) (by decide) (by decide)
      (by decide)
  · refine span_xor3_not_of_distinct_at (k := k) (n := n) (p := 6) (q := 11) (r := 25) (w := 6)
      (hwq := Or.inr rfl) (hnw := h)
      (by omega) (by omega) (by omega) (by omega) (by decide) (by omega) (by decide) (by decide)
      (by decide)

/-- **`σ₀`**, with witnesses `2^7` (for `n ≠ 7`) and `2^18` (for `n = 7`). -/
theorem smallSigma0_not_span {k n : Nat} (hk : 26 ≤ k) (_hn : 1 ≤ n) (_hnk : n < k) :
    ¬ Span2 (k := k) n (smallSigma0Val k) := by
  by_cases h : n = 7
  · refine span_xor2rot_shr_not_of_distinct_at (k := k) (n := n) (p := 7) (q := 18) (r := 3) (w := 18)
      (hwq := Or.inl rfl) (hnw := by omega)
      (by omega) (by omega) (by omega) (by decide) (by omega) (by decide) (by decide) (by decide)
  · refine span_xor2rot_shr_not_of_distinct_at (k := k) (n := n) (p := 7) (q := 18) (r := 3) (w := 7)
      (hwq := Or.inr rfl) (hnw := h)
      (by omega) (by omega) (by omega) (by decide) (by omega) (by decide) (by decide) (by decide)

/-- **`σ₁`**, with witnesses `2^17` (for `n ≠ 17`) and `2^19` (for `n = 17`). -/
theorem smallSigma1_not_span {k n : Nat} (hk : 26 ≤ k) (_hn : 1 ≤ n) (_hnk : n < k) :
    ¬ Span2 (k := k) n (smallSigma1Val k) := by
  by_cases h : n = 17
  · refine span_xor2rot_shr_not_of_distinct_at (k := k) (n := n) (p := 17) (q := 19) (r := 10) (w := 19)
      (hwq := Or.inl rfl) (hnw := by omega)
      (by omega) (by omega) (by omega) (by decide) (by omega) (by decide) (by decide) (by decide)
  · refine span_xor2rot_shr_not_of_distinct_at (k := k) (n := n) (p := 17) (q := 19) (r := 10) (w := 17)
      (hwq := Or.inr rfl) (hnw := h)
      (by omega) (by omega) (by omega) (by decide) (by omega) (by decide) (by decide) (by decide)

/-- **The obstruction stated at the corpus's own width, `k = 32`**: no split of the 32-bit word
    serves any of the four composites. This is `bigSigma0_not_span` and its three siblings with
    `k = 32`, i.e. the FIPS regime, with the hypotheses discharged. -/
theorem composites_not_span_at_32 :
    ¬ Span2 (k := 32) 7 (bigSigma0Val 32)
      ∧ ¬ Span2 (k := 32) 11 (bigSigma1Val 32)
      ∧ ¬ Span2 (k := 32) 18 (smallSigma0Val 32)
      ∧ ¬ Span2 (k := 32) 19 (smallSigma1Val 32) := by
  refine ⟨bigSigma0_not_span (by omega) (by omega) (by omega),
    bigSigma1_not_span (by omega) (by omega) (by omega),
    smallSigma0_not_span (by omega) (by omega) (by omega),
    smallSigma1_not_span (by omega) (by omega) (by omega)⟩

/-! ## §4 The cost count under the halves reading, against AR-15's round cost -/

/-- An operation-count vector in AR-15's `RoundCost` order,
    `(mul, add, sub, neg, cast, channel, xor, and, not, shift, or)`. `RingCost`'s body counts are
    **cited** from `work/proofs/AlgebraCostRound.md:117-137` (`ringMul_body_ops`,
    `ringUpper_body_ops`, `rotr_body_ops`, `shrx_body_ops`, `choose_body_ops`,
    `majority_body_ops`, `roundCost_fields`, `schedStepCost_fields`), each read off a canonical
    Lean definition body there; every arithmetic statement below is proved in this file. A count
    is not a measurement (AGENTS.md §5, obligation 10 does not apply). -/
structure OpCount where
  mul : Nat
  add : Nat
  sub : Nat
  neg : Nat
  cast : Nat
  channel : Nat
  xor : Nat
  ann : Nat
  nnot : Nat
  shift : Nat
  orr : Nat
  deriving Repr, DecidableEq

/-- Componentwise sum. -/
def OpCount.sum (a b : OpCount) : OpCount :=
  ⟨a.mul + b.mul, a.add + b.add, a.sub + b.sub, a.neg + b.neg, a.cast + b.cast,
    a.channel + b.channel, a.xor + b.xor, a.ann + b.ann, a.nnot + b.nnot,
    a.shift + b.shift, a.orr + b.orr⟩

/-- Componentwise multiple. -/
def OpCount.rep (n : Nat) (a : OpCount) : OpCount :=
  ⟨n * a.mul, n * a.add, n * a.sub, n * a.neg, n * a.cast, n * a.channel, n * a.xor,
    n * a.ann, n * a.nnot, n * a.shift, n * a.orr⟩

/-- `ringMul`'s body count, as AR-15 records it: five multiplications, two additions. The dial
    term `σ·z.2·w.2` is one of the five and is paid even at `σ = −1`. -/
def ringMulOps : OpCount := ⟨5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩

/-- `ringUpper`'s body count: one multiplication, one addition, one channel read. -/
def ringUpperOps : OpCount := ⟨1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0⟩

/-- The canonical `rotr` body: two shifts and one disjunction. -/
def rotrBodyOps : OpCount := ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1⟩

/-- The canonical `shrx` body: one shift. -/
def shrBodyOps : OpCount := ⟨0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0⟩

/-- `choose`: two ANDs, one NOT, two XORs. -/
def chooseOps : OpCount := ⟨0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0⟩

/-- `majority`: three ANDs, two XORs. -/
def majorityOps : OpCount := ⟨0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0⟩

/-- One `Word` addition. -/
def wordAddOps : OpCount := ⟨0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0⟩

/-- One exclusive-or: no ring operation at all, under every reading. -/
def xorOps : OpCount := ⟨0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0⟩

/-- **The bit-halves pairing's own cost**: `x &&& (2^n − 1)` and `x >>> n`, one AND and one
    shift, the mask `2^n − 1` being an amount constant compiled once per amount. This replaces
    the canonical `rotr` body's two shifts and one disjunction. -/
def pairingOps : OpCount := ⟨0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0⟩

/-- **AR-15's rotation reading** `ringUpper r (ringMul 1 (x,0) (2^n, 2^{k−n}))` plus the canonical
    `rotr` body. -/
def c5RotationOps : OpCount := OpCount.sum (OpCount.sum ringMulOps ringUpperOps) rotrBodyOps

/-- **The halves reading of one rotation term**: the pairing, then one `ringMul (−1)` by the forced
    multiplier, read at the first slot. No channel read (`ellipticRead` is slot 1), and **no
    residual subtraction** — the pairing's read is exact. -/
def halvesRotationOps : OpCount := OpCount.sum pairingOps ringMulOps

/-- One round under AR-15's reading: six rotations, `choose`, `majority`, four `⊕`s of the two
    `Σ` layers, seven word additions. -/
def c5RoundOps : OpCount :=
  OpCount.sum (OpCount.rep 6 c5RotationOps) chooseOps |> (fun x => OpCount.sum x majorityOps)
    |> (fun x => OpCount.sum x (OpCount.rep 7 wordAddOps))
    |> (fun x => OpCount.sum x (OpCount.rep 4 xorOps))

/-- One round under the halves reading, with the same non-rotation body operations. -/
def halvesRoundOps : OpCount :=
  OpCount.sum (OpCount.rep 6 halvesRotationOps) chooseOps |> (fun x => OpCount.sum x majorityOps)
    |> (fun x => OpCount.sum x (OpCount.rep 7 wordAddOps))
    |> (fun x => OpCount.sum x (OpCount.rep 4 xorOps))

/-- One schedule step under AR-15's reading: four rotations, two shifts, four `⊕`s, three
    additions. -/
def c5StepOps : OpCount :=
  OpCount.sum (OpCount.rep 4 c5RotationOps) (OpCount.rep 2 shrBodyOps)
    |> (fun x => OpCount.sum x (OpCount.rep 4 xorOps))
    |> (fun x => OpCount.sum x (OpCount.rep 3 wordAddOps))

/-- One schedule step under the halves reading. The two `SHR` terms are **their own cell's second
    coordinate**, a projection, so they cost no ring operation under either reading. -/
def halvesStepOps : OpCount :=
  OpCount.sum (OpCount.rep 4 halvesRotationOps) (OpCount.rep 2 shrBodyOps)
    |> (fun x => OpCount.sum x (OpCount.rep 4 xorOps))
    |> (fun x => OpCount.sum x (OpCount.rep 3 wordAddOps))

/-- One composite of three rotations (`Σ₀`, `Σ₁`) under the halves reading: three pairings, three
    forced multiplications, two `⊕`s. -/
def halvesBigSigmaOps : OpCount := OpCount.sum (OpCount.rep 3 halvesRotationOps) (OpCount.rep 2 xorOps)

/-- One composite of two rotations and a shift (`σ₀`, `σ₁`) under the halves reading; the `SHR`
    term is a projection. -/
def halvesSmallSigmaOps : OpCount := OpCount.sum (OpCount.rep 2 halvesRotationOps) (OpCount.rep 2 xorOps)

/-- **The same model, applied to AR-15's reading, reproduces AR-15's published round vector
    exactly** — so the difference below is like for like. -/
theorem c5RoundOps_eq_ar15 : c5RoundOps = ⟨36, 25, 0, 0, 0, 6, 8, 5, 1, 12, 6⟩ := by decide

/-- **The halves reading of one round**: thirty multiplications, nineteen additions, no channel
    reads. -/
theorem halvesRoundOps_eq : halvesRoundOps = ⟨30, 19, 0, 0, 0, 0, 8, 11, 1, 6, 0⟩ := by decide

/-- **The comparison.** Ring-side operations per round (`mul + add + channel`): sixty-seven under
    AR-15's reading, forty-nine under the halves reading — a saving of eighteen per round, of
    which six are the eliminated `ringUpper` reads and twelve the eliminated per-rotation
    `mul`/`add`. The `⊕` layer is unchanged at eight per round and is the residual this file does
    not remove. -/
theorem round_ring_side_comparison :
    c5RoundOps.mul + c5RoundOps.add + c5RoundOps.channel = 67
      ∧ halvesRoundOps.mul + halvesRoundOps.add + halvesRoundOps.channel = 49
      ∧ c5RoundOps.xor = 8 ∧ halvesRoundOps.xor = 8 := by decide

/-- **The word-side price of the pairing**: six ANDs of masks, against the six disjunctions and six
    shifts the halves reading no longer needs. -/
theorem round_word_side_comparison :
    c5RoundOps.ann = 5 ∧ halvesRoundOps.ann = 11
      ∧ c5RoundOps.orr = 6 ∧ halvesRoundOps.orr = 0
      ∧ c5RoundOps.shift = 12 ∧ halvesRoundOps.shift = 6 := by decide

/-- **The per-term count**, the unit the round figure is built from: five multiplications, two
    additions, one AND, one shift — against AR-15's six, three, one channel, two shifts, one
    disjunction. No subtraction in either: the pairing's read needs no residual. -/
theorem rotation_term_comparison :
    halvesRotationOps.mul = 5 ∧ halvesRotationOps.add = 2 ∧ halvesRotationOps.channel = 0
      ∧ halvesRotationOps.sub = 0
      ∧ c5RotationOps.mul = 6 ∧ c5RotationOps.add = 3 ∧ c5RotationOps.channel = 1 := by decide

/-- **One composite** under the halves reading, with and without the `SHR` term. -/
theorem composite_cost :
    halvesBigSigmaOps = ⟨15, 6, 0, 0, 0, 0, 2, 3, 0, 3, 0⟩
      ∧ halvesSmallSigmaOps = ⟨10, 4, 0, 0, 0, 0, 2, 2, 0, 2, 0⟩ := by decide

/-- **The schedule step.** Our reconstruction of AR-15's step figure under its own reading gives
    `add = 15` where AR-15 publishes `16` (`AlgebraCostRound.md:183`); we do not use the step
    figure. Under the halves reading the step is `⟨20, 11, 0, 0, 0, 0, 4, 4, 0, 6, 0⟩`. -/
theorem schedule_step_costs :
    c5StepOps = ⟨24, 15, 0, 0, 0, 4, 4, 0, 0, 10, 4⟩
      ∧ halvesStepOps = ⟨20, 11, 0, 0, 0, 0, 4, 4, 0, 6, 0⟩ := by decide

/-! ## §5 The axiom cone -/

#print axioms ringMul_cell_eq
#print axioms ellipticRead_eq
#print axioms denNat_rotrVal
#print axioms slot1_exact
#print axioms rotrRead_eq_denNat
#print axioms rotrRead_eq_ellipticRead
#print axioms slot2_exact
#print axioms revHalves_eq_sub
#print axioms revHalves_of_loBits_zero
#print axioms halves_recover
#print axioms rotrVal_eq_rotrBv
#print axioms rotrVal_lt
#print axioms shrVal_eq_shrxBv
#print axioms two_pow_ge_two
#print axioms two_pow_lt_two_pow
#print axioms two_pow_le_two_pow
#print axioms halves_degenerate
#print axioms halves_read_is_identity_above_width
#print axioms halves_multiplier_norm_form
#print axioms halves_unitary_of
#print axioms halves_not_unitary_of
#print axioms sha256_amounts_unitary
#print axioms two_pow_mod_two
#print axioms two_pow_mod_two'
#print axioms xor_mod_two
#print axioms xor3_mod_two
#print axioms loBits_pow2
#print axioms hiBits_pow2
#print axioms halves_pow2_even
#print axioms rotrVal_eq_hi_lo
#print axioms rotrVal_pow2_mod_two
#print axioms shrVal_pow2_mod_two
#print axioms rotrVal_pow2_self
#print axioms shrVal_pow2_self
#print axioms zmod_ne_of_parity
#print axioms not_span_of_evenCell_oddVal
#print axioms span_rotr_iff
#print axioms span_shr_iff
#print axioms shrRead_is_second_coordinate
#print axioms slot1_table_width4
#print axioms slot2_table_width4
#print axioms slot1_table_width2
#print axioms halves_width4_witness
#print axioms span_rotr_width4_own
#print axioms span_rotr_width4_not
#print axioms span_shr_width4_own
#print axioms span_shr_width4_not
#print axioms xor3_rot_odd
#print axioms xor3_rot_shr_odd
#print axioms xor3_rot_odd'
#print axioms xor3_rot_shr_odd'
#print axioms span_xor3_not_of_distinct_at
#print axioms span_xor2rot_shr_not_of_distinct_at
#print axioms span_xor3_not_of_distinct
#print axioms span_xor2rot_shr_not_of_distinct
#print axioms shrVal_lt
#print axioms rotrRead_val
#print axioms bigSigma0_reads
#print axioms bigSigma1_reads
#print axioms smallSigma0_reads
#print axioms smallSigma1_reads
#print axioms bigSigma0_not_span
#print axioms bigSigma1_not_span
#print axioms smallSigma0_not_span
#print axioms smallSigma1_not_span
#print axioms composites_not_span_at_32
#print axioms c5RoundOps_eq_ar15
#print axioms halvesRoundOps_eq
#print axioms round_ring_side_comparison
#print axioms round_word_side_comparison
#print axioms rotation_term_comparison
#print axioms composite_cost
#print axioms schedule_step_costs

end AlgebraSigmaHalves
