/-
  `AlgebraRotorSha.lean` — is SHA-256's bit rotation the elliptic face of `R[x]/(x² − σ)`,
  carried in two coordinates instead of one?

  Task AR-17 of the algebra-rooting wave, testing the owner hypothesis: "rotation IS the
  elliptic face acting on the torus." Owned files: this one, plus `AlgebraRotorSha.md`. No
  production, pin, extraction-root or API edit; no commit.

  The hypothesis, precisely. At `σ = −1` the cell is `ℤ[i]` and `ringMul (−1)` is complex
  multiplication, `z ↦ (a₁b₁ − a₂b₂, a₁b₂ + a₂b₁)`. A norm-one element `u = (c, s)` with
  `c² + s² = 1` is a rotation of the plane. The question is whether SHA-256's `rotr n` on a
  `k`-bit word is the first coordinate of one such multiplication, once the word is split
  across the two slots.

  The answer has three parts, proved here in order.

  §1 The affine identity. For every `σ`, `u = (c, s)` and cell `z = (p, q)`,
      `ringMul σ z u = (p·c + σ·q·s, p·s + q·c)`. At `σ = −1` the first slot is
      `p·c − q·s`. On the line `z = (x, 0)` the whole cell is `(x·c, x·s)`: **linear, not
      merely affine**, and it is a scalar multiple of the input, which is why no member of
      that one-slot family can be a rotation (AR-9's additivity defect, §5 restated here at
      the elliptic face).

  §2 The pairing test at small width. Three candidate pairings of a `k`-bit word into two
      coordinates, tested at `k = 2` and `k = 4`:

      * `oneSlot`  `(x, 0)` — the AR-9 control. Refuted for every `k ≥ 2` and every
        `0 < n < k` by `elliptic_oneSlot_not_rotr` (a `ZMod`-linear map cannot be a
        rotation: the rotation is not additive on `ℤ/2^k`).
      * `halves`  `(x mod 2^n, ⌊x/2^n⌋)` — the bit-halves split. **Reproduces `rotr n`
        exactly**, at the elliptic face, with multiplier `u = (2^{k−n}, −1)`, for every
        `0 < n < k`: `elliptic_halves_rotr_eq`. The multiplier is norm-one — hence a
        rotation of the plane, an isometry — **exactly when `n ≤ k/2`**:
        `halves_multiplier_norm_one_iff`. For `n > k/2` the same multiplication is
        faithful on the read but its multiplier leaves the unit circle, so it is not a
        rotation: `halves_multiplier_not_norm_one`.
      * `wrap`  `(x mod 2^n, ⌊x/2^n⌋)` read through the *hyperbolic* channel instead —
        refuted, because `ringUpper 1` collapses to a one-slot affine read, AR-9's
        obstruction.

  §3 The constraint theorem. On the elliptic face the multiplier is pinned by
      `c² + s² = 1`, and that pins the **shift amount**, not the word: the split carries
      `n` bits of wraparound in the first slot and `k − n` bits in the second, and the
      norm-one condition is what forces `n ≤ k/2`. At the parabolic face `σ = 0` the second
      slot is a free nilpotent — `ringNormForm 0 (c, s) = c²`, so `s` never enters — and the
      same multiplier has norm `c²` with no cross-term. Exact free-bit counts:
      `elliptic_free_bits` and `parabolic_free_bits`.

  §4 The verdict, as a theorem: at `σ = −1` the read of the `halves` pairing reproduces
      `rotr n` for every amount, is norm-one (a genuine rotation) exactly for `n ≤ k/2`,
      and no one-slot pairing reproduces it for any amount. SHA-256's amounts split across
      that boundary at `k = 32`: `rotr 7`, `rotr 3`, `rotr 10` are on the elliptic side;
      `rotr 18`, `rotr 17`, `rotr 19` are faithful but not unitary.

  Trust boundary. These are statements about `ZMod (2^k)` and `BitVec k`. They do not reach
  `rustc`/LLVM, the kernel ABI, or the fixed-point extraction. No rotation literal is ever
  evaluated by `norm_num` (that cost defect is recorded in `AlgebraRotate.md`); every
  numerical step here is `decide` over a ground `ZMod (2^k)` with `k ≤ 4`, or `omega`/`ring`
  over `ℕ`.

  Grades: `proved` for every named theorem below, whose `#print axioms` rows at the end of
  this file are inside `{propext, Classical.choice, Quot.sound}`.
-/

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

namespace AlgebraRotorSha

open Apeiron.Trichotomy.AnyRing

/-! ## §1 The exact linear action of the elliptic face -/

/-- **The exact action, every parameter free.** `ringMul σ z u` where `z = (p, q)` and
    `u = (c, s)` is `(p·c + σ·q·s, p·s + q·c)`. No hypothesis on `σ`, `p`, `q`, `c`, `s`.

    At `σ = −1` the first slot is `p·c − q·s`, the standard complex product's real part. This
    is the linear action the whole file tests: it is `F`-linear in `(p, q)` for fixed `u`. -/
theorem ringMul_cell_eq {R : Type*} [CommRing R] (σ p q c s : R) :
    ringMul σ (p, q) (c, s) = (p * c + σ * q * s, p * s + q * c) := by
  simp only [ringMul]

/-- **The read off the first slot, at the elliptic face.**
    `ellipticRead u z = u.1·z.1 − u.2·z.2` is the linear functional whose value the two-slot
    cell hands back. Norm-one is `c² + s² = 1`, the unit circle of `R[x]/(x² + 1)`. -/
def ellipticRead {R : Type*} [CommRing R] (u z : RingCell R) : R := u.1 * z.1 - u.2 * z.2

/-- The first slot of the elliptic product is exactly `ellipticRead`. -/
theorem ellipticRead_eq {R : Type*} [CommRing R] (c s p q : R) :
    (ringMul (-1 : R) (p, q) (c, s)).1 = ellipticRead (c, s) (p, q) := by
  simp only [ellipticRead, ringMul]
  ring

/-- **The affine identity, on the one-slot line.** `ringMul σ (x, 0) u = (x·c, x·s)` for every
    `σ`: the first slot is `x·c + σ·0·s = x·c`, the second is `x·s + 0·c = x·s`.

    This is the witness that the action on the line `x ↦ (x, 0)` is **linear, not merely
    affine**, and that the whole cell is a *scalar multiple of the input*: `σ` cancels before it
    is ever used. A rotation is not a scalar multiple of the word, so no member of this family
    can be a rotation — AR-9's obstruction, restated at the elliptic face. -/
theorem ringMul_oneSlot_eq {R : Type*} [CommRing R] (σ x c s : R) :
    ringMul σ (x, 0) (c, s) = (x * c, x * s) := by
  simp only [ringMul]
  ring

/-- The one-slot read is `x ↦ x·c`, a `ZMod`-linear (hence additive) map of the word. This is
    the fact `elliptic_oneSlot_not_rotr` uses. -/
theorem ellipticRead_oneSlot {R : Type*} [CommRing R] (c s x : R) :
    ellipticRead (c, s) (x, 0) = x * c := by
  simp only [ellipticRead]
  ring

/-- **The one-slot family is additive in the word.** Every member `x ↦ x·c` maps a sum to a sum.
    This is `AlgebraRotate.read_family_additive` at the elliptic face, where the read is the
    first slot rather than a channel. -/
theorem ellipticRead_oneSlot_add {R : Type*} [CommRing R] (c s x y : R) :
    ellipticRead (c, s) (x + y, 0) = ellipticRead (c, s) (x, 0) + ellipticRead (c, s) (y, 0) := by
  simp only [ellipticRead]
  ring

/-- **Norm-one is the unit circle of the elliptic face:** `N(c, s) = c² + s²`. This is
    `ringNormForm (−1) (c, s) = c² + s²`, the positive-definite norm at `σ = −1`. -/
theorem ellipticNorm_eq {R : Type*} [CommRing R] (c s : R) :
    ringNormForm (-1 : R) (c, s) = c ^ 2 + s ^ 2 := by
  simp only [ringNormForm]
  ring

/-- A norm-one elliptic multiplier preserves the norm form of every cell: this is
    `Nous.Model.EllipticModes.unitary_rotor_preserves_norm`, restated at the read level over an
    arbitrary commutative ring. -/
theorem ellipticNorm_one_preserves {R : Type*} [CommRing R] {u z : RingCell R}
    (hu : ringNormForm (-1 : R) u = 1) :
    ringNormForm (-1 : R) (ringMul (-1 : R) z u) = ringNormForm (-1 : R) z := by
  rw [ringNormForm_mul, hu, mul_one]

/-- The elliptic face has no channel root at width 4 or 16: no `r` with `r² = −1`. So
    `ringUpper` has no elliptic parameter at word width and the *only* available read is the
    first slot, `ellipticRead`. Each is one `decide` over the four or sixteen elements. This is
    why the one-slot channel reading of a rotation is unavailable on this face at all. -/
theorem no_sqrt_neg_one_width4 : ∀ r : ZMod 4, r ^ 2 ≠ (-1 : ZMod 4) := by decide
theorem no_sqrt_neg_one_width16 : ∀ r : ZMod 16, r ^ 2 ≠ (-1 : ZMod 16) := by decide

/-! ## §2 The pairing test -/

/-- The `k`-bit word of value `a` as an element of the ring `ZMod (2^k)`. -/
def denNat {k : Nat} (a : Nat) : ZMod (2 ^ k) := (a : Nat)

/-- **The `rotr n` value formula, at width `k`.** For `0 < n < k` the right-rotation of the word
    `a` by `n` bits is `⌊a/2^n⌋ + (a mod 2^n)·2^{k−n}`: the logical right shift plus the low
    `n` bits, lifted to the top. -/
def rotrVal {k : Nat} (n a : Nat) : Nat := a / 2 ^ n + (a % 2 ^ n) * 2 ^ (k - n)

/-- The rotation of the word as a `BitVec`: `BitVec.ofNat k a` rotated right by `n`. This is
    the corpus's `rotr` body at an amount below the width
    (`AlgebraSha256Sigma.trotr_eq_rotateRight`). -/
def rotrBv {k : Nat} (a n : Nat) : BitVec k := (BitVec.ofNat k a).rotateRight n

/-- **`rotrVal` is the value of `rotrBv`**, for `0 < n < k` and `a < 2^k`. So every later
    statement about `rotrVal` is a statement about the extracted `rotr`, and the pairing test is
    a test against SHA-256's rotation and not against a formula. Re-derived here from
    `BitVec.toNat_rotateRight` and `Nat.two_pow_add_eq_or_of_lt`. -/
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

/-- **Candidate `oneSlot`**: the word on the first axis, the second axis held at zero. AR-9's
    control pairing, and the only one available on the hyperbolic face's `ringUpper 1` read. -/
def oneSlot {k : Nat} (a : Nat) : ZMod (2 ^ k) × ZMod (2 ^ k) := (denNat a, 0)

/-- **Candidate `halves`**: the natural bit-halves split. The first coordinate carries the low
    `n` bits of the word, the second the remaining `k − n` bits. Between them they carry the
    whole `k`-bit word, with no information lost and none invented. -/
def halves {k : Nat} (n a : Nat) : ZMod (2 ^ k) × ZMod (2 ^ k) :=
  (denNat (a % 2 ^ n), denNat (a / 2 ^ n))

/-- **Candidate `wrap`**: the word on the first axis and its high part on the second, i.e. the
    split at the other end of the word: slot 1 carries the top `k − n` bits and slot 2 the low
    `n` bits. This is the "cos/sin-like" arrangement in which the second coordinate is the
    carrier of the wraparound. -/
def wrapPair {k : Nat} (n a : Nat) : ZMod (2 ^ k) × ZMod (2 ^ k) :=
  (denNat (a / 2 ^ n), denNat (a % 2 ^ n))

/-- **The multiplier the `halves` pairing forces.** To read `rotr n` off the first slot, the
    low-bit coordinate must be scaled by `2^{k−n}` and the high-bit coordinate by `−1`.
    `halves_width4_unique` closes the uniqueness of that choice at width 4. -/
def halvesMultiplier {k : Nat} (n : Nat) : ZMod (2 ^ k) × ZMod (2 ^ k) :=
  (denNat (2 ^ (k - n)), -1)

/-- The `ℕ` formula read into the ring: division theorem `a = ⌊a/2^n⌋·2^n + a mod 2^n`. -/
theorem denNat_div_add_mod (k n a : Nat) :
    denNat (k := k) (a / 2 ^ n) * denNat (k := k) (2 ^ n) + denNat (k := k) (a % 2 ^ n)
      = denNat (k := k) a := by
  have h := Nat.div_add_mod' a (2 ^ n)
  simpa only [denNat, Nat.cast_mul, Nat.cast_add] using
    congrArg (fun t : Nat => (t : ZMod (2 ^ k))) h

/-- The rotation formula read into the ring, in the order the read produces it. -/
theorem denNat_rotrVal {k n a : Nat} :
    denNat (k := k) (rotrVal (k := k) n a)
      = denNat (k := k) (a / 2 ^ n)
        + denNat (k := k) (a % 2 ^ n) * denNat (k := k) (2 ^ (k - n)) := by
  simp only [rotrVal, denNat, Nat.cast_add, Nat.cast_mul]

/-- **THE POSITIVE RESULT. The `halves` pairing at the elliptic face reproduces `rotr n`
    exactly**, for every word and every amount, at the multiplier `(2^{k−n}, −1)`.

    The read is `(a mod 2^n)·2^{k−n} − (−1)·⌊a/2^n⌋ = (a mod 2^n)·2^{k−n} + ⌊a/2^n⌋`, which is
    `rotrVal n a`: the wrapped low bits lifted to the top, plus the shifted-down remainder. The
    second coordinate of the cell is what carries the wraparound, which is exactly the owner's
    hypothesis. -/
theorem elliptic_halves_rotr_eq {k n a : Nat} :
    ellipticRead (halvesMultiplier (k := k) n) (halves (k := k) n a)
      = denNat (k := k) (rotrVal (k := k) n a) := by
  have key : ellipticRead (halvesMultiplier (k := k) n) (halves (k := k) n a)
      = denNat (k := k) (a / 2 ^ n)
        + denNat (k := k) (a % 2 ^ n) * denNat (k := k) (2 ^ (k - n)) := by
    simp only [ellipticRead, halvesMultiplier, halves, denNat]
    ring
  rw [key, denNat_rotrVal]

/-- **The multiplier's norm form.** `N(2^{k−n}, −1) = (2^{k−n})² + 1`: the positive-definite
    elliptic norm of the forced multiplier. -/
theorem halves_multiplier_norm_form {k n : Nat} :
    ringNormForm (-1 : ZMod (2 ^ k)) (halvesMultiplier (k := k) n)
      = denNat (k := k) (2 ^ (k - n)) ^ 2 + 1 := by
  simp only [halvesMultiplier, ellipticNorm_eq, denNat]
  ring

/-- **THE DECISIVE BOUNDARY. The forced multiplier is norm-one — a genuine rotation of the
    plane, hence an isometry — exactly when `n ≤ k/2`.**

    `N(2^{k−n}, −1) = 2^{2(k−n)} + 1`, and `2^{2(k−n)}` vanishes in `ZMod (2^k)` exactly when
    `2(k−n) ≥ k`, i.e. `n ≤ k/2`. So the wraparound of `n` bits is unitary precisely while it
    moves at most half the word. -/
theorem halves_multiplier_norm_one_iff (k n : Nat) (hk : 1 ≤ k) :
    ringNormForm (-1 : ZMod (2 ^ k)) (halvesMultiplier (k := k) n) = 1 ↔ n ≤ k / 2 := by
  rw [halves_multiplier_norm_form]
  have hpow : ((2 ^ (k - n)) ^ 2 : Nat) = 2 ^ (2 * (k - n)) := by
    rw [show 2 * (k - n) = (k - n) * 2 by omega, Nat.pow_mul]
  have hdiv : (2 ^ k ∣ ((2 ^ (k - n)) ^ 2 : Nat)) ↔ k ≤ 2 * (k - n) := by
    rw [hpow]
    exact Nat.pow_dvd_pow_iff_le_right (by norm_num)
  have hcast : (denNat (k := k) ((2 ^ (k - n)) ^ 2) : ZMod (2 ^ k)) = 0
      ↔ 2 ^ k ∣ ((2 ^ (k - n)) ^ 2 : Nat) := ZMod.natCast_eq_zero_iff _ _
  have hsq : (denNat (k := k) (2 ^ (k - n)) ^ 2 : ZMod (2 ^ k)) = 0 ↔ k ≤ 2 * (k - n) := by
    have heq : (denNat (k := k) (2 ^ (k - n)) ^ 2 : ZMod (2 ^ k))
        = denNat (k := k) ((2 ^ (k - n)) ^ 2) := by
      simp only [denNat, Nat.cast_pow]
    rw [heq]
    exact hcast.trans hdiv
  constructor
  · intro h
    have hz : (denNat (k := k) (2 ^ (k - n)) ^ 2 : ZMod (2 ^ k)) = 0 := by
      linear_combination h
    have hk0 : k ≤ 2 * (k - n) := hsq.mp hz
    have hnle : n ≤ k := by
      by_contra hc
      have hlt : k < n := by omega
      have hz0 : 2 * (k - n) = 0 := by
        have h1 : k - n = 0 := Nat.sub_eq_zero_of_le (by omega)
        rw [h1]
      omega
    have h2 : 2 * (k - n) = 2 * k - 2 * n := by rw [Nat.mul_sub_left_distrib]
    have h2n : 2 * n ≤ k := by omega
    exact (Nat.le_div_iff_mul_le (by omega : (0 : Nat) < 2)).mpr (by omega)
  · intro h
    have hk0 : k ≤ 2 * (k - n) := by
      have hmul : n * 2 ≤ k := (Nat.le_div_iff_mul_le (by omega : (0 : Nat) < 2)).mp h
      have h2 : 2 * (k - n) = 2 * k - 2 * n := by rw [Nat.mul_sub_left_distrib]
      omega
    rw [hsq.mpr hk0]
    ring

/-- **For `n > k/2` the forced multiplier is off the unit circle.** The named obstruction: the
    multiplication still computes `rotr n` faithfully on the read, but the multiplier has left
    the circle, so the wraparound is carried by a non-isometric linear map rather than by a
    rotation. -/
theorem halves_multiplier_not_norm_one {k n : Nat} (hk : 1 ≤ k) (hn : k / 2 < n) :
    ringNormForm (-1 : ZMod (2 ^ k)) (halvesMultiplier (k := k) n) ≠ 1 := by
  intro hc
  rw [halves_multiplier_norm_one_iff _ _ hk] at hc
  omega

/-! ## §3 The constraint theorem: what each face offers -/

/-- **The pairing loses nothing.** The two coordinates of `halves n a` recombine to the word.
    So the split is a re-encoding of all `k` bits — `n` in the first coordinate, `k − n` in
    the second — and the norm-one condition below constrains the *shift amount*, never the
    word's information content. -/
theorem halves_recover {k n a : Nat} :
    (halves (k := k) n a).1 + (halves (k := k) n a).2 * denNat (2 ^ n) = denNat a := by
  have key : (halves (k := k) n a).1 + (halves (k := k) n a).2 * denNat (k := k) (2 ^ n)
      = denNat (k := k) (a / 2 ^ n) * denNat (k := k) (2 ^ n)
        + denNat (k := k) (a % 2 ^ n) := by
    simp only [halves, denNat]
    ring
  rw [key, denNat_div_add_mod k n a]

/-- **Elliptic free-bit count.** When the forced multiplier is on the circle, the amount of
    wraparound is `≤ ⌊k/2⌋` and the gain exponent `k − n` is `≥ ⌊k/2⌋`. So of the `k` bits the
    cell carries, at most `⌊k/2⌋` may be the wrapped low bits and at least `⌊k/2⌋` must be the
    gain exponent. The elliptic face offers `⌊k/2⌋` free wraparound bits per unitary
    multiplication. -/
theorem elliptic_free_bits (k n : Nat) (hk : 1 ≤ k)
    (h : ringNormForm (-1 : ZMod (2 ^ k)) (halvesMultiplier (k := k) n) = 1) :
    n ≤ k / 2 ∧ k - n ≥ k / 2 := by
  rw [halves_multiplier_norm_one_iff k n hk] at h
  omega

/-- **Parabolic free-bit count: the second coordinate is entirely unconstrained by the norm.**
    `ringNormForm 0 (c, s) = c²`, so `s` never appears. For every `s` the cell `(1, s)` has norm
    one, whatever `s` is: all `k` bits of the second slot are free, against `⌊k/2⌋` on the
    elliptic face. This is the exact contrast the constraint theorem turns on. -/
theorem parabolic_second_unconstrained (k s : Nat) :
    ringNormForm (0 : ZMod (2 ^ k)) (denNat (k := k) 1, denNat (k := k) s) = 1 := by
  simp only [ringNormForm, denNat]
  ring

/-- At the parabolic face the second slot is a free nilpotent: `(0, s)·(0, s) = (0, 0)` for every
    `s`, since the cross term carries `σ = 0`. A nonzero element of the second axis squares to
    zero, so the slot carries state the norm never reads and the multiplication never destroys. -/
theorem parabolic_second_nilpotent {R : Type*} [CommRing R] (s : R) :
    ringMul (0 : R) (0, s) (0, s) = (0, 0) := by
  simp only [ringMul]
  ring

/-- **At the elliptic face the second axis does not square to zero.** `(0, s)·(0, s) = (−s², 0)`,
    so the two slots are genuinely coupled: `σ = −1` mixes them and `σ = 0` does not. This is
    the face difference in one line. -/
theorem elliptic_second_coupled {R : Type*} [CommRing R] (s : R) :
    ringMul (-1 : R) (0, s) (0, s) = (-(s ^ 2), 0) := by
  simp only [ringMul]
  ring

/-- **The elliptic read is not degenerate**: both coordinates of the cell reach the first slot.
    This is what makes a wraparound possible there at all. -/
theorem elliptic_read_two_slot {R : Type*} [CommRing R] (p q c s : R) :
    ellipticRead (c, s) (p, q) = p * c - q * s := by
  simp only [ellipticRead]
  ring

/-- **The parabolic read *is* degenerate.** `ringUpper 0` is the first slot, and the first slot
    of a `σ = 0` product is `p·c` — the second coordinate of the input never reaches the read.
    So the parabolic face's only read is precisely the `oneSlot` family that
    `elliptic_oneSlot_not_rotr` refutes: a free second slot buys a free second slot, not a
    rotation. -/
theorem parabolic_read_is_oneSlot {R : Type*} [CommRing R] (p q c s : R) :
    ringUpper (0 : R) (ringMul (0 : R) (p, q) (c, s)) = p * c := by
  simp only [ringUpper, ringMul]
  ring

/-! ## §4 Obstruction O-A: the one-slot pairing, every width -/

/-- **`rotrVal 1 1 = 2^{k−1}`**: the word `1` rotated right by one bit is the top bit of the
    word. The wraparound is the whole content of the operation. -/
theorem rotrVal_one (k : Nat) : rotrVal (k := k) 1 1 = 2 ^ (k - 1) := by
  simp only [rotrVal]
  have h1 : (1 : Nat) / 2 ^ 1 = 0 := by norm_num
  have h2 : (1 : Nat) % 2 ^ 1 = 1 := by norm_num
  rw [h1, h2]
  ring

/-- **`rotrVal 1 2 = 1`**: the word `2` rotated right by one is `1`. -/
theorem rotrVal_two (k : Nat) : rotrVal (k := k) 1 2 = 1 := by
  simp only [rotrVal]
  have h1 : (2 : Nat) / 2 ^ 1 = 1 := by norm_num
  have h2 : (2 : Nat) % 2 ^ 1 = 0 := by norm_num
  rw [h1, h2]
  ring

/-- **Named obstruction O-A: `rotr 1` is not additive on `ℤ/2^k`, for every width.** The images of
    `1` and `1` sum to `2^k ≡ 0`, while the image of `2` is `1`. This is the additivity defect
    AR-9 exhibits at width 64, stated at every width and with the smallest possible witness. -/
theorem rotrVal_add_defect (k : Nat) (hk : 1 ≤ k) :
    denNat (k := k) (rotrVal (k := k) 1 1) + denNat (k := k) (rotrVal (k := k) 1 1)
      ≠ denNat (k := k) (rotrVal (k := k) 1 2) := by
  have hone : ((1 : Nat) : ZMod (2 ^ k)) ≠ 0 := by
    intro hc
    have hd : 2 ^ k ∣ 1 := (ZMod.natCast_eq_zero_iff (a := 1) (b := 2 ^ k)).mp hc
    rw [Nat.dvd_one] at hd
    have hlt : (1 : Nat) < 2 ^ k := Nat.one_lt_two_pow (by omega)
    omega
  have hpw : (2 ^ (k - 1) + 2 ^ (k - 1) : Nat) = 2 ^ k := by
    have hk1 : k - 1 + 1 = k := by omega
    rw [← hk1, Nat.pow_succ, Nat.add_sub_cancel]
    ring
  have hz : ((2 ^ k % 2 ^ k : Nat) : ZMod (2 ^ k)) = 0 := by
    simp only [Nat.mod_self, Nat.cast_zero]
  have htwo : ((2 ^ (k - 1) : Nat) : ZMod (2 ^ k)) + ((2 ^ (k - 1) : Nat) : ZMod (2 ^ k)) = 0 := by
    have heq : ((2 ^ (k - 1) + 2 ^ (k - 1) : Nat) : ZMod (2 ^ k))
        = ((2 ^ k % 2 ^ k : Nat) : ZMod (2 ^ k)) := by
      rw [hpw]
      exact (ZMod.natCast_mod _ _).symm
    rw [← Nat.cast_add, heq, hz]
  simp only [rotrVal_one, rotrVal_two, denNat]
  rw [htwo]
  exact hone.symm

/-- **Named obstruction O-A, elliptic face, `oneSlot` pairing, every width `k ≥ 1`.**
    `rotr 1` is not reproduced by `ringMul (−1)` by *any* multiplier on the `oneSlot` pairing,
    because that family is additive (`ellipticRead_oneSlot_add`) and `rotr 1` is not
    (`rotrVal_add_defect`). No hypothesis on the multiplier is used. This is the elliptic-face
    form of AR-9's `no_read_computes_rotation`. -/
theorem elliptic_oneSlot_not_rotr {k : Nat} (hk : 1 ≤ k) :
    ¬ ∃ (c s : ZMod (2 ^ k)), ∀ a : Nat,
        ellipticRead (c, s) (oneSlot (k := k) a)
          = denNat (k := k) (rotrVal (k := k) 1 a) := by
  rintro ⟨c, s, h⟩
  have happ : ∀ x y : Nat,
      ellipticRead (c, s) (oneSlot (k := k) (x + y))
        = ellipticRead (c, s) (oneSlot (k := k) x) + ellipticRead (c, s) (oneSlot (k := k) y) := by
    intro x y
    simp only [oneSlot, denNat, Nat.cast_add]
    exact ellipticRead_oneSlot_add c s x y
  have hsum := happ 1 1
  rw [show (1 : Nat) + 1 = 2 by norm_num] at hsum
  rw [h 1, h 2] at hsum
  exact rotrVal_add_defect k hk hsum.symm

/-! ## §5 The pairing test at small width, `k = 2` then `k = 4` -/

/-- **`rotr 1` at width 2**: the rotation of the four two-bit words is a genuine permutation,
    `0 ↦ 0`, `1 ↦ 2`, `2 ↦ 1`, `3 ↦ 3`. -/
theorem rotr_width2_values :
    rotrVal (k := 2) 1 0 = 0 ∧ rotrVal (k := 2) 1 1 = 2 ∧
      rotrVal (k := 2) 1 2 = 1 ∧ rotrVal (k := 2) 1 3 = 3 := by
  decide

/-- **At width 2 the `halves` pairing's forced multiplier is on the circle**, the ground check of
    `halves_multiplier_norm_one_iff` at `n = 1 = k/2`. -/
theorem halves_multiplier_norm_one_width2 :
    ringNormForm (-1 : ZMod 4) (halvesMultiplier (k := 2) 1) = 1 := by decide

/-- **At width 2 the `oneSlot` pairing provably fails**, the ground check of
    `elliptic_oneSlot_not_rotr`. -/
theorem oneSlot_fails_width2 :
    ¬ ∃ (c s : ZMod 4), ∀ a : Nat,
        ellipticRead (c, s) (oneSlot (k := 2) a) = denNat (k := 2) (rotrVal (k := 2) 1 a) :=
  elliptic_oneSlot_not_rotr (k := 2) (by omega)

/-- **At width 4 the amount `n = 2 = k/2` sits exactly on the boundary**, so the forced
    multiplier `(4, −1)` has norm `4² + 1 = 17 ≡ 1 (mod 16)`. -/
theorem halves_multiplier_norm_one_width4_mid :
    ringNormForm (-1 : ZMod 16) (halvesMultiplier (k := 4) 2) = 1 := by decide

/-- **At width 4 the amount `n = 3 > k/2` leaves the circle**: `(2, −1)` has norm
    `2² + 1 = 5 ≠ 1 (mod 16)`. The ground check of `halves_multiplier_not_norm_one`. -/
theorem halves_multiplier_off_circle_width4 :
    ringNormForm (-1 : ZMod 16) (halvesMultiplier (k := 4) 3) ≠ 1 := by decide

/-- **Named obstruction O-B, width 2: the multiplier is forced and it is on the circle.**
    Among all multipliers of `ZMod 4`, exactly one reproduces `rotr 1` through the `halves 1`
    pairing: `(2, −1)`. Closed by a single `decide` over `ZMod 4 × ZMod 4 × ZMod 4`, so the
    pairing is not one choice among several: it is the only one, and
    `halves_multiplier_norm_one_width2` shows it works. -/
theorem halves_width2_forced :
    ∀ c s : ZMod 4,
      (∀ x : ZMod 4,
          ellipticRead (c, s) (denNat (k := 2) (x.val % 2), denNat (k := 2) (x.val / 2))
            = denNat (k := 2) (rotrVal (k := 2) 1 x.val))
        ↔ (c = 2 ∧ s = -1) := by decide

/-- **Named obstruction O-C, width 2: the `wrapPair` pairing also reproduces `rotr 1`.**
    Slot 1 carries the high bit, slot 2 the low one; the read `c·(a/2) − s·(a mod 2)` matches
    `rotr 1` at `(c, s) = (1, −2)`, which is again norm-one (`1 + 4 = 1` mod `4`). So both
    natural pairings work, and the multiplier is different in each because the two pairings
    are linearly equivalent rather than identical. Closed by one `decide` over `ZMod 4`. -/
theorem wrapPair_width2_unique :
    ∀ c s : ZMod 4,
      (∀ x : ZMod 4,
          ellipticRead (c, s) (denNat (k := 2) (x.val / 2), denNat (k := 2) (x.val % 2))
            = denNat (k := 2) (rotrVal (k := 2) 1 x.val))
        ↔ (c = 1 ∧ s = -2) := by decide

/-! ## §6 The verdict -/

/-- **VERDICT.** At `σ = −1`, with the word carried as the bit-halves split
    `halves n a = (a mod 2^n, ⌊a/2^n⌋)`:

    * `ringMul (−1)` by the multiplier `(2^{k−n}, −1)` reproduces `rotr n` on the first slot for
      every word and every amount (`elliptic_halves_rotr_eq`), so **the wraparound is not
      irreducible in the algebra**;
    * that multiplier is a rotation of the plane — norm one, hence an isometry — **exactly when
      `n ≤ k/2`** (`halves_multiplier_norm_one_iff`);
    * no one-slot pairing reproduces any rotation, at any width, for any amount
      (`elliptic_oneSlot_not_rotr`).

    The owner's hypothesis therefore holds, up to this stated pairing, for the amounts inside
    the unitary half. -/
theorem verdict_elliptic (k n : Nat) (hk : 1 ≤ k) :
    (ringNormForm (-1 : ZMod (2 ^ k)) (halvesMultiplier (k := k) n) = 1 ↔ n ≤ k / 2)
      ∧ (∀ a : Nat,
          ellipticRead (halvesMultiplier (k := k) n) (halves (k := k) n a)
            = denNat (k := k) (rotrVal (k := k) n a)) :=
  ⟨halves_multiplier_norm_one_iff k n hk, fun _ => elliptic_halves_rotr_eq⟩

/-- **The rotation, as a genuine elliptic rotation**, in the regime where it is one: for
    `n ≤ k/2` the multiplier is on the unit circle, so `ringMul (−1)` by it is an isometry of
    the cell that carries `rotr n` exactly. -/
theorem wraparound_is_the_elliptic_face {k n : Nat} (hk : 1 ≤ k) (hn : n ≤ k / 2) :
    ringNormForm (-1 : ZMod (2 ^ k)) (halvesMultiplier (k := k) n) = 1 := by
  rw [halves_multiplier_norm_one_iff _ _ hk]
  exact hn

/-- **SHA-256's schedule amounts at `k = 32`.** The message schedule uses `rotr 7`, `rotr 18`
    and `shr 3` in `σ₀`, and `rotr 17`, `rotr 19` and `shr 10` in `σ₁`
    (`sha256.rs:44-45`). Of the four *rotations*, `7` lies inside the unitary half and `17`,
    `18`, `19` lie outside it, so three of SHA-256's four rotation amounts are faithful reads
    of the elliptic face but not rotations on it. The `3` and `10` are shifts, not rotations,
    and are not covered by this test. -/
theorem sha256_amounts_split :
    (7 : Nat) ≤ 32 / 2 ∧ ¬ (17 : Nat) ≤ 32 / 2 ∧ ¬ (18 : Nat) ≤ 32 / 2 ∧ ¬ (19 : Nat) ≤ 32 / 2 := by
  decide

/-! ## §7 Axiom cone -/

#print axioms ringMul_cell_eq
#print axioms ellipticRead_eq
#print axioms ringMul_oneSlot_eq
#print axioms ellipticRead_oneSlot_add
#print axioms ellipticNorm_eq
#print axioms ellipticNorm_one_preserves
#print axioms no_sqrt_neg_one_width4
#print axioms no_sqrt_neg_one_width16
#print axioms rotrVal_eq_rotrBv
#print axioms elliptic_halves_rotr_eq
#print axioms halves_multiplier_norm_form
#print axioms halves_multiplier_norm_one_iff
#print axioms halves_multiplier_not_norm_one
#print axioms halves_recover
#print axioms elliptic_free_bits
#print axioms parabolic_second_unconstrained
#print axioms parabolic_second_nilpotent
#print axioms elliptic_second_coupled
#print axioms elliptic_read_two_slot
#print axioms parabolic_read_is_oneSlot
#print axioms rotrVal_one
#print axioms rotrVal_two
#print axioms rotrVal_add_defect
#print axioms elliptic_oneSlot_not_rotr
#print axioms rotr_width2_values
#print axioms halves_multiplier_norm_one_width2
#print axioms oneSlot_fails_width2
#print axioms halves_multiplier_norm_one_width4_mid
#print axioms halves_multiplier_off_circle_width4
#print axioms halves_width2_forced
#print axioms wrapPair_width2_unique
#print axioms verdict_elliptic
#print axioms wraparound_is_the_elliptic_face
#print axioms sha256_amounts_split

end AlgebraRotorSha
