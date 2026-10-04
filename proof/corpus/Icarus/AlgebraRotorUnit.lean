/-
  `AlgebraRotorUnit.lean` — is the `n > k/2` failure of AR-17's elliptic face a genuine
  boundary, or a property of the two pairings AR-17 named?

  Task AR-25 of the algebra-rooting wave. Owned files: this one, plus `AlgebraRotorUnit.md`.
  No production, pin, extraction-root or API edit; no commit.

  AR-17 (`work/proofs/AlgebraRotorSha.lean`) proved that for the bit-halves pairing
  `halves n a = (a mod 2^n, ⌊a/2^n⌋)` the multiplier reproducing `rotr n` is unique and
  norm-one exactly when `n ≤ k/2`, and left open whether another pairing carries a
  unit-circle multiplier at `n > k/2`. This file answers that.

  §0 The corpus interface, re-derived: `ringMul`, the elliptic read, the norm form, and the
     identification of `rotrVal` with the extracted `BitVec.rotateRight`.

  §1 **The pairing classes, stated.** No bijection `BitVec k → ZMod (2^k)²` exists (`2^k ≠
     4^k`), so a pairing is a map and each class is an explicit formula:

       * `affPair (α β γ δ) a = (α·a + β, γ·a + δ)` — **Class A**, affine in the word.
       * `halfPair (α β γ δ) n a = (α·lo + β·hi, γ·lo + δ·hi)` with `lo = a mod 2^n`,
         `hi = ⌊a/2^n⌋` — **Class B**, linear in the bit-halves split.

     AR-17's `oneSlot` is Class A at `(1,0,0,0)`; its `halves` is Class B at `(1,0,0,1)`; its
     `wrapPair` is Class B at `(0,1,1,0)`.

  §2 **The forced-multiplier method, generalised** (item 2). `halfPair_reproduces_iff` proves
     the reproducing multipliers of a Class B pairing are *exactly* the solutions of the
     linear system `c·α − s·γ = 2^(k−n)`, `c·β − s·δ = 1`. When `D = β·γ − α·δ` is a unit
     the family is a singleton and the member is named (`halfPair_unique_of_unit`);
     `halfPair_family_add` shows the family is closed under adding a kernel vector, so a
     nontrivial kernel yields a genuinely larger family, exhibited at width 2 by
     `singular_family_width2`. AR-17's `halves_width2_forced` is the `(1,0,0,1)` instance of
     the singleton case, recovered by `halvesMultiplier_is_forced`.

  §3 **Class A is refuted at every width and amount** — `affine_pairing_no_rotr`, obstruction
     **O-A′**, which strictly generalises AR-17's `elliptic_oneSlot_not_rotr`.

  §4 **The answer for Class B (item 1): the `k/2` boundary is a property of the pairing's
     scale, not of the amount** — `verdict_unitary_beyond_half`. For `σ` a unit the
     scaled-halves pairing `scaledPair n σ a = (a mod 2^n, σ·⌊a/2^n⌋)` forces the multiplier
     `(2^(k−n), −σ⁻¹)` and its norm-one condition is `scaled_unitary_iff_square`. The
     explicit unit `σ = 1 + 2^(2(k−n)−1)` makes that square, giving `scaled_unitary_of_amount`:
     for `k ≤ 4·(k−n) − 2` the pairing `scaledPair n (1 + 2^(2(k−n)−1))` with multiplier
     `(2^(k−n), 2^(2(k−n)−1) − 1)` reproduces `rotr n` **and** is norm one. At `k = 32` this
     holds for every `n ≤ 23`, so all four SHA-256 rotation amounts are unitary under a
     Class B pairing (`sha256_amounts_unitary`).

  §5 **Ground checks at `k = 4` and `k = 8` only.** Width 4, the smallest width, has no Class B
     unitary multiplier at `n = 3 > k/2` (`no_unitary_amount3_width4`); width 8 has one at
     `n = 5 > k/2` (`unitary_amount5_width8`), and `scaled_read_width8` / `scaled_norm_width8`
     re-derive the width-8 witness on all `2^8` words by `decide`. `halves_multiplier_off_circle_width8`
     confirms AR-17's boundary at width 8. The reflection amount `n = k − 1` fails the square
     condition at both widths (`no_quarter_square_width8`); `reflecting_amount_no_square` is
     the named general statement and is **not** proved here.

  §6 **The cos/sin carrier** (item 3). AR-17 left uncovered "a genuine cos/sin table (a
     non-re-encoding carrier)". The corpus supplies the carrier: `Nous.Model.RopeOrbit` gives
     `rhoMul` on `ℝ × ℝ` and the unit circle `ρ(θ) = (cos θ, sin θ)`. With the pairing
     `ropeCarrier k a = ρ(2π·a/2^k)`, `rope_no_rotr_mult` (**O-E**) shows the multiplier must
     be the identity (`rope_at_zero_forces_identity`: `a = 0` sits at the identity of `T`) and
     then fails at `a = 1` by injectivity of `cos` on `[0, π]`. So the cos/sin carrier is not
     an escape from O-B: it is a stronger obstruction, refuting every nonzero `rotr n`.
     Transporting the carrier to `ZMod 2^k` needs an element of order `2^k` in the `ZMod`
     unit circle; that premise is stated, not proved — see the residual obligations.

  Trust boundary. Statements are about `ZMod (2^k)`, `BitVec k` and (in §6 only) `ℝ`. They do
  not reach `rustc`/LLVM, the kernel ABI, or the fixed-point extraction. No security,
  collision-resistance or constant-time claim. No exhaustive `decide` above `k = 8`; no
  `native_decide`; no raised `maxRecDepth`; no `maxHeartbeats 0`.

  Grades: `proved` for every theorem below whose `#print axioms` row is inside
  `{propext, Classical.choice, Quot.sound}`.
-/

import Apeiron.Trichotomy.AnyRing
import Nous.Model.RopeOrbit

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
-- Port (W2-T3): the `hk`/`hn`/`hσ` binders below are carried in the fixed pairing-class and
-- width statements and are deliberately not re-cited in every body.
set_option linter.unusedVariables false

namespace AlgebraRotorUnit

open Apeiron.Trichotomy.AnyRing

/-! ## §0 The corpus interface and the read, re-derived -/

/-- **The exact action of `ringMul`, every parameter free.** `ringMul σ z u` at `z = (p, q)`,
    `u = (c, s)` is `(p·c + σ·q·s, p·s + q·c)`. -/
theorem ringMul_cell_eq {R : Type*} [CommRing R] (σ p q c s : R) :
    ringMul σ (p, q) (c, s) = (p * c + σ * q * s, p * s + q * c) := by
  simp only [ringMul]

/-- The read off the first slot at the elliptic face `σ = −1`. -/
def ellipticRead {R : Type*} [CommRing R] (u z : RingCell R) : R := u.1 * z.1 - u.2 * z.2

theorem ellipticRead_eq {R : Type*} [CommRing R] (c s p q : R) :
    (ringMul (-1 : R) (p, q) (c, s)).1 = ellipticRead (c, s) (p, q) := by
  simp only [ellipticRead, ringMul]
  ring

/-- **Norm one is the unit circle of the elliptic face**: `N(c, s) = c² + s²`. -/
theorem ellipticNorm_eq {R : Type*} [CommRing R] (c s : R) :
    ringNormForm (-1 : R) (c, s) = c ^ 2 + s ^ 2 := by
  simp only [ringNormForm]
  ring

/-- The `k`-bit word of value `a` as an element of `ZMod (2^k)`. -/
def denNat {k : Nat} (a : Nat) : ZMod (2 ^ k) := (a : Nat)

/-- The `rotr n` value formula at width `k`. -/
def rotrVal {k : Nat} (n a : Nat) : Nat := a / 2 ^ n + (a % 2 ^ n) * 2 ^ (k - n)

/-- The rotation of the word as a `BitVec`. -/
def rotrBv {k : Nat} (a n : Nat) : BitVec k := (BitVec.ofNat k a).rotateRight n

/-- **`rotrVal` is the value of `rotrBv`**, so every statement below is about SHA-256's
    `rotr` and not about a formula chosen to fit. -/
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

/-- The rotation formula read into the ring, in the order the read produces it. -/
theorem denNat_rotrVal {k n a : Nat} :
    denNat (k := k) (rotrVal (k := k) n a)
      = denNat (k := k) (a / 2 ^ n)
        + denNat (k := k) (a % 2 ^ n) * denNat (k := k) (2 ^ (k - n)) := by
  simp only [rotrVal, denNat, Nat.cast_add, Nat.cast_mul]

/-! ## §1 The two pairing classes -/

/-- **Class A, affine in the word.** The instance `(1,0,0,0)` is AR-17's `oneSlot`; the
    instance `(1,0,1,0)` is the modular sum `ringUpper 1 (a, 0) = a`, the hyperbolic read. -/
def affPair {k : Nat} (α β γ δ : ZMod (2 ^ k)) (a : Nat) :
    ZMod (2 ^ k) × ZMod (2 ^ k) :=
  (α * denNat a + β, γ * denNat a + δ)

/-- **Class B, linear in the bit-halves split**, coordinate order `(low, high)`:
    `halfPair (α β γ δ) n a = (α·lo + β·hi, γ·lo + δ·hi)` with `lo = a mod 2^n`,
    `hi = ⌊a/2^n⌋`. AR-17's `halves` is `(α,β,γ,δ) = (1,0,0,1)`. -/
def halfPair {k n : Nat} (α β γ δ : ZMod (2 ^ k)) (a : Nat) :
    ZMod (2 ^ k) × ZMod (2 ^ k) :=
  (α * denNat (a % 2 ^ n) + β * denNat (a / 2 ^ n),
    γ * denNat (a % 2 ^ n) + δ * denNat (a / 2 ^ n))

/-! ## §2 The forced-multiplier method, generalised to Class B -/

/-- **The read through a Class B pairing is a two-term linear form in the halves.** -/
theorem halfPair_read_eq {k n : Nat} (α β γ δ c s : ZMod (2 ^ k)) (a : Nat) :
    ellipticRead (c, s) (halfPair (k := k) (n := n) α β γ δ a)
      = (c * α - s * γ) * denNat (a % 2 ^ n) + (c * β - s * δ) * denNat (a / 2 ^ n) := by
  simp only [ellipticRead, halfPair, denNat]
  ring

theorem halfPair_at_one {k n : Nat} (hn : 0 < n) (α β γ δ : ZMod (2 ^ k)) :
    halfPair (k := k) (n := n) α β γ δ 1 = (α, γ) := by
  have h1 : (1 : Nat) % 2 ^ n = 1 := by rw [Nat.mod_eq_of_lt (Nat.one_lt_two_pow (by omega))]
  have h2 : (1 : Nat) / 2 ^ n = 0 := by rw [Nat.div_eq_of_lt (Nat.one_lt_two_pow (by omega))]
  simp [halfPair, h1, h2, denNat]

theorem halfPair_at_pow {k n : Nat} (α β γ δ : ZMod (2 ^ k)) :
    halfPair (k := k) (n := n) α β γ δ (2 ^ n) = (β, δ) := by
  have h1 : (2 ^ n : Nat) % 2 ^ n = 0 := Nat.mod_self _
  have h2 : (2 ^ n : Nat) / 2 ^ n = 1 := by rw [Nat.div_self (Nat.two_pow_pos n)]
  simp [halfPair, h1, h2, denNat]

theorem rotrVal_one' {k n : Nat} (hn : 0 < n) : rotrVal (k := k) n 1 = 2 ^ (k - n) := by
  have h1 : (1 : Nat) % 2 ^ n = 1 := by rw [Nat.mod_eq_of_lt (Nat.one_lt_two_pow (by omega))]
  have h2 : (1 : Nat) / 2 ^ n = 0 := by rw [Nat.div_eq_of_lt (Nat.one_lt_two_pow (by omega))]
  simp only [rotrVal, h1, h2, Nat.mul_one, Nat.one_mul, Nat.zero_add]

theorem rotrVal_pow' {k n : Nat} : rotrVal (k := k) n (2 ^ n) = 1 := by
  have h1 : (2 ^ n : Nat) % 2 ^ n = 0 := Nat.mod_self _
  have h2 : (2 ^ n : Nat) / 2 ^ n = 1 := by rw [Nat.div_self (Nat.two_pow_pos n)]
  simp only [rotrVal, h1, h2, Nat.zero_mul, Nat.zero_add]

/-- **The general forced-multiplier theorem (item 2).** For a Class B pairing and an amount
    `0 < n`, the multipliers reproducing `rotr n` are *exactly* the solutions of the linear
    system

      `c·α − s·γ = 2^(k−n)`,  `c·β − s·δ = 1`.

    The two test words are `a = 1` and `a = 2^n`, on which `(lo, hi)` is `(1, 0)` and
    `(0, 1)`, so the read is exactly the `lo`-coefficient and the `hi`-coefficient. This is
    AR-17's forced-multiplier argument for the bit-halves pairing, with the pairing's four
    scale factors as free parameters. -/
theorem halfPair_reproduces_iff {k n : Nat} (hk : 1 ≤ k) (hn : 0 < n)
    (α β γ δ c s : ZMod (2 ^ k)) :
    (∀ a : Nat, ellipticRead (c, s) (halfPair (k := k) (n := n) α β γ δ a)
        = denNat (rotrVal (k := k) n a))
      ↔ c * α - s * γ = denNat (2 ^ (k - n)) ∧ c * β - s * δ = 1 := by
  constructor
  · intro h
    have h1 := h 1
    rw [halfPair_at_one hn, ellipticRead, rotrVal_one' hn] at h1
    have h2 := h (2 ^ n)
    rw [halfPair_at_pow, ellipticRead, rotrVal_pow'] at h2
    exact ⟨h1, by simpa [denNat] using h2⟩
  · rintro ⟨hα, hβ⟩
    intro a
    rw [halfPair_read_eq, hα, hβ, denNat_rotrVal]
    ring

/-- **Uniqueness when the determinant is invertible (item 2, the `D` a unit case).** If
    `D = β·γ − α·δ` is invertible, the reproducing multiplier of a Class B pairing is a
    singleton: the difference of two solutions is in the kernel of the system map, and the
    two equations force `D` to annihilate it. The *value* is then read off the system, which
    is `scaled_forced` for the family carrying the refutation. -/
theorem halfPair_unique {k n : Nat} (α β γ δ c s c' s' : ZMod (2 ^ k))
    (hD : ((β * γ - α * δ) : ZMod (2 ^ k))⁻¹ * (β * γ - α * δ) = 1)
    (hsol : c * α - s * γ = denNat (2 ^ (k - n)) ∧ c * β - s * δ = 1)
    (hsol' : c' * α - s' * γ = denNat (2 ^ (k - n)) ∧ c' * β - s' * δ = 1) :
    c = c' ∧ s = s' := by
  have he1 : (c - c') * α - (s - s') * γ = 0 := by
    have e : (c - c') * α - (s - s') * γ = (c * α - s * γ) - (c' * α - s' * γ) := by ring
    rw [e, hsol.1, hsol'.1, sub_self]
  have he2 : (c - c') * β - (s - s') * δ = 0 := by
    have e : (c - c') * β - (s - s') * δ = (c * β - s * δ) - (c' * β - s' * δ) := by ring
    rw [e, hsol.2, hsol'.2, sub_self]
  have hA : (s - s') * γ = (c - c') * α := by
    calc (s - s') * γ = (c - c') * α - ((c - c') * α - (s - s') * γ) := by ring
      _ = (c - c') * α - 0 := by rw [he1]
      _ = (c - c') * α := by ring
  have hB : (s - s') * δ = (c - c') * β := by
    calc (s - s') * δ = (c - c') * β - ((c - c') * β - (s - s') * δ) := by ring
      _ = (c - c') * β - 0 := by rw [he2]
      _ = (c - c') * β := by ring
  have key : (β * γ - α * δ) * (c - c') = 0 := by
    calc (β * γ - α * δ) * (c - c') = γ * (β * (c - c')) - δ * (α * (c - c')) := by ring
      _ = γ * ((s - s') * δ) - δ * ((s - s') * γ) := by
          rw [mul_comm β (c - c'), mul_comm α (c - c'), hB, hA]
      _ = 0 := by ring
  have key2 : (β * γ - α * δ) * (s - s') = 0 := by
    calc (β * γ - α * δ) * (s - s') = β * ((s - s') * γ) - α * ((s - s') * δ) := by ring
      _ = β * ((c - c') * α) - α * ((c - c') * β) := by rw [hA, hB]
      _ = 0 := by ring
  constructor
  · have e : c - c' = 0 := by
      calc c - c' = (β * γ - α * δ)⁻¹ * ((β * γ - α * δ) * (c - c')) := by
            rw [← mul_assoc, hD, one_mul]
        _ = (β * γ - α * δ)⁻¹ * 0 := by rw [key]
        _ = 0 := mul_zero _
    exact sub_eq_zero.mp e
  · have e : s - s' = 0 := by
      calc s - s' = (β * γ - α * δ)⁻¹ * ((β * γ - α * δ) * (s - s')) := by
            rw [← mul_assoc, hD, one_mul]
        _ = (β * γ - α * δ)⁻¹ * 0 := by rw [key2]
        _ = 0 := mul_zero _
    exact sub_eq_zero.mp e

/-- **The family is closed under adding a kernel vector.** `(e, f)` in the kernel of the
    system map means `e·α − f·γ = 0` and `e·β − f·δ = 0`; adding it to a solution gives
    another solution. So a nontrivial kernel yields a genuinely larger family of
    reproducing multipliers — the "exhibit the family" half of item 2. -/
theorem halfPair_family_add {k n : Nat} (α β γ δ c s e f : ZMod (2 ^ k))
    (hker : e * α - f * γ = 0 ∧ e * β - f * δ = 0)
    (hsol : c * α - s * γ = denNat (2 ^ (k - n)) ∧ c * β - s * δ = 1) :
    (c + e) * α - (s + f) * γ = denNat (2 ^ (k - n)) ∧ (c + e) * β - (s + f) * δ = 1 := by
  obtain ⟨h1, h2⟩ := hker
  obtain ⟨h3, h4⟩ := hsol
  have e1 : e * α - f * γ = 0 := h1
  have e2 : e * β - f * δ = 0 := h2
  constructor
  · calc (c + e) * α - (s + f) * γ = (c * α - s * γ) + (e * α - f * γ) := by ring
    _ = denNat (2 ^ (k - n)) + 0 := by rw [h3, e1]
    _ = denNat (2 ^ (k - n)) := by ring
  · calc (c + e) * β - (s + f) * δ = (c * β - s * δ) + (e * β - f * δ) := by ring
    _ = 1 + 0 := by rw [h4, e2]
    _ = 1 := by ring

/-! ## §3 Class A: obstruction O-A′ -/

/-- **The read through a Class A pairing is affine in the word.** -/
theorem affine_read_eq {k : Nat} (α β γ δ c s : ZMod (2 ^ k)) (a : Nat) :
    ellipticRead (c, s) (affPair (k := k) α β γ δ a)
      = (c * α - s * γ) * denNat a + (c * β - s * δ) := by
  simp only [ellipticRead, affPair, denNat]
  ring

/-- **Named obstruction O-A′: no affine pairing reproduces `rotr n`, for any width `k ≥ 1`
    and any amount `0 < n < k`, with *any* multiplier — norm-one or not.**

    The read of a Class A pairing is affine, so it is `A·a + B`. At `a = 0` this forces
    `B = 0`; at `a = 1` it forces `A = 2^(k−n)`; at `a = 2^n` it then gives
    `A·2^n = 2^k ≡ 0`, while `rotrVal n (2^n) = 1`. This strictly contains AR-17's
    `elliptic_oneSlot_not_rotr` (the instance `α = 1, β = γ = δ = 0`) and AR-9's
    `no_read_computes_rotation` on the hyperbolic read (the instance `α = γ = 1`). -/
theorem affine_pairing_no_rotr {k n : Nat} (hk : 1 ≤ k) (hn : 0 < n) (hnk : n < k) :
    ¬ ∃ (α β γ δ c s : ZMod (2 ^ k)),
        ∀ a : Nat, ellipticRead (c, s) (affPair (k := k) α β γ δ a)
          = denNat (rotrVal (k := k) n a) := by
  rintro ⟨α, β, γ, δ, c, s, h⟩
  have hr : ∀ (a : Nat), ellipticRead (c, s) (affPair (k := k) α β γ δ a)
      = (c * α - s * γ) * denNat a + (c * β - s * δ) := by
    intro a
    exact affine_read_eq α β γ δ c s a
  have hz0 : rotrVal (k := k) n 0 = 0 := by simp [rotrVal]
  have hd0 : (denNat (k := k) (0 : Nat) : ZMod (2 ^ k)) = 0 := by
    simp only [denNat, Nat.cast_zero]
  have hd1 : (denNat (k := k) (1 : Nat) : ZMod (2 ^ k)) = 1 := by
    simp only [denNat, Nat.cast_one]
  have hzero : c * β - s * δ = 0 := by
    have h0 := h 0
    rw [hr, hz0, hd0] at h0
    simpa using h0
  have hA : c * α - s * γ = denNat (2 ^ (k - n)) := by
    have h1 := h 1
    rw [hr, rotrVal_one' hn, hd1, hzero] at h1
    simpa using h1
  have hz : (denNat (k := k) (2 ^ (k - n))) * denNat (k := k) (2 ^ n) = 0 := by
    simp only [denNat]
    rw [← Nat.cast_mul]
    have h : 2 ^ (k - n) * 2 ^ n = 2 ^ k := by
      rw [← Nat.pow_add]
      congr 1
      omega
    rw [h, ZMod.natCast_self]
  have hne : (denNat (1 : Nat) : ZMod (2 ^ k)) ≠ 0 := by
    intro h1
    exact Nat.not_dvd_of_pos_of_lt (by norm_num) (Nat.one_lt_two_pow (by omega : k ≠ 0))
      ((ZMod.natCast_eq_zero_iff (1 : ℕ) (2 ^ k)).mp (by simpa [denNat] using h1))
  have h2 := h (2 ^ n)
  rw [hr, rotrVal_pow', hd1, hA, hzero] at h2
  rw [hz, zero_add] at h2
  exact hne (by simpa [denNat] using h2.symm)

/-! ## §4 Class B: the boundary is a property of the pairing scale -/

/-- **The scaled-halves pairing**: the low window verbatim, the high window scaled by a unit
    `σ`. Both coordinates are pure bit windows of the word, so the pairing carries the `k`
    bits without mixing them and without precomputing the rotation. -/
def scaledPair {k n : Nat} (σ : ZMod (2 ^ k)) (a : Nat) :
    ZMod (2 ^ k) × ZMod (2 ^ k) :=
  halfPair (k := k) (n := n) 1 0 0 σ a

/-- The pairing's first coordinate is the low window unchanged. -/
theorem scaledPair_lo {k n : Nat} (σ : ZMod (2 ^ k)) (a : Nat) :
    (scaledPair (k := k) (n := n) σ a).1 = denNat (a % 2 ^ n) := by
  simp [scaledPair, halfPair]

/-- **The scaled-halves pairing reproduces `rotr n` at the forced multiplier
    `(2^(k−n), −σ⁻¹)`, for every unit `σ`.** This is `halfPair_reproduces_iff` at
    `(α,β,γ,δ) = (1,0,0,σ)`: the two scale factors enter separately, `σ` controlling the
    second coordinate of the multiplier only. -/
theorem scaled_read_rotr {k n : Nat} (σ : ZMod (2 ^ k)) (hσ : σ⁻¹ * σ = 1) (a : Nat) :
    ellipticRead (denNat (2 ^ (k - n)), -(σ⁻¹)) (scaledPair (k := k) (n := n) σ a)
      = denNat (rotrVal (k := k) n a) := by
  simp only [scaledPair, halfPair_read_eq]
  have e1 : (denNat (2 ^ (k - n)) * 1 - -(σ⁻¹) * 0) * denNat (a % 2 ^ n)
      + (denNat (2 ^ (k - n)) * 0 - -(σ⁻¹) * σ) * denNat (a / 2 ^ n)
      = denNat (a / 2 ^ n) + denNat (a % 2 ^ n) * denNat (2 ^ (k - n)) := by
    have hneg : -(σ⁻¹) * σ = -(σ⁻¹ * σ) := by ring
    rw [hneg, hσ]
    ring
  rw [e1, denNat_rotrVal]

/-- **The multiplier a scaled-halves pairing forces.** Through
    `scaledPair n σ = halfPair 1 0 0 σ`, the system of `halfPair_reproduces_iff` is
    `c = 2^(k−n)` and `s·σ = −1`, so the multiplier is `(2^(k−n), −σ⁻¹)` — and the gain
    exponent `k−n` is untouched by the pairing's scale. -/
theorem scaled_forced {k n : Nat} (hn : 0 < n) (σ : ZMod (2 ^ k)) (hσ : σ⁻¹ * σ = 1)
    (c s : ZMod (2 ^ k))
    (hsol : c * 1 - s * 0 = denNat (2 ^ (k - n)) ∧ c * 0 - s * σ = 1) :
    c = denNat (2 ^ (k - n)) ∧ s = -(σ⁻¹) := by
  obtain ⟨h1, h2⟩ := hsol
  have hsn : s * σ = -1 := by
    calc s * σ = -(c * 0 - s * σ) := by ring
      _ = -1 := congrArg Neg.neg h2
  have hsn' : s * σ * σ⁻¹ = -(σ⁻¹) := by
    calc s * σ * σ⁻¹ = (-1) * σ⁻¹ := by rw [hsn]
      _ = -(σ⁻¹) := by ring
  constructor
  · calc c = c * 1 - s * 0 := by ring
    _ = denNat (2 ^ (k - n)) := h1
  · calc s = (s * σ) * σ⁻¹ := by rw [mul_assoc, mul_comm σ σ⁻¹, hσ, mul_one]
    _ = -(σ⁻¹) := hsn'

/-- **The exact norm-one criterion for the forced multiplier of a scaled-halves pairing**:
    it is a rotation of the plane (norm one, hence an isometry) exactly when
    `1 − 2^(2(k−n))` is a square in `ZMod (2^k)`, with the square root being `σ⁻¹`. -/
theorem scaled_unitary_iff_square {k n : Nat} (σ : ZMod (2 ^ k)) (hσ : σ⁻¹ * σ = 1) :
    ringNormForm (-1 : ZMod (2 ^ k)) (denNat (2 ^ (k - n)), -(σ⁻¹)) = 1
      ↔ σ⁻¹ * σ⁻¹ = 1 - denNat (2 ^ (k - n)) * denNat (2 ^ (k - n)) := by
  have hpow : (-(σ⁻¹)) ^ 2 = σ⁻¹ * σ⁻¹ := by ring
  rw [ellipticNorm_eq, hpow]
  constructor
  · intro h
    linear_combination h
  · intro h
    linear_combination h

/-- The scale that makes the circle close: `σ = 1 + 2^(2(k−n)−1)`. Its inverse is
    `1 − 2^(2(k−n)−1)` whenever `2^k ∣ 2^(4(k−n) − 2)`, and the forced multiplier is
    `(2^(k−n), 2^(2(k−n)−1) − 1)`, whose norm is `2^(4(k−n)−2) + 1 ≡ 1`. -/
def scaledSigma (k n : Nat) : ZMod (2 ^ k) := denNat (1 + 2 ^ (2 * (k - n) - 1))

/-- The forced multiplier at `σ = scaledSigma`. -/
def scaledUnitary (k n : Nat) : ZMod (2 ^ k) × ZMod (2 ^ k) :=
  (denNat (2 ^ (k - n)), denNat (2 ^ (2 * (k - n) - 1) - 1))

theorem scaled_sigma_inv {k n : Nat} (hk : 1 ≤ k) (hnk : n < k) (h4 : k ≤ 4 * (k - n) - 2) :
    scaledSigma k n * (denNat (1 : Nat) - denNat (2 ^ (2 * (k - n) - 1))) = 1 := by
  have hkn : 0 < k - n := by omega
  have h2m : k ≤ 2 * (2 * (k - n) - 1) := by omega
  have hd : (denNat (2 ^ (2 * (k - n) - 1)) : ZMod (2 ^ k)) ^ 2 = 0 := by
    simp only [denNat]
    rw [← Nat.cast_pow, ZMod.natCast_eq_zero_iff, ← pow_mul]
    rw [Nat.pow_dvd_pow_iff_le_right (show (1 : Nat) < 2 by norm_num)]
    omega
  have hsum : scaledSigma k n
      = 1 + denNat (2 ^ (2 * (k - n) - 1)) := by
    simp only [scaledSigma, denNat, Nat.cast_add, Nat.cast_one]
  have hd1 : (denNat (1 : Nat) : ZMod (2 ^ k)) = 1 := by
    simp only [denNat, Nat.cast_one]
  rw [hsum, hd1]
  have e2 : (denNat (k := k) (2 ^ (2 * (k - n) - 1))) * denNat (k := k) (2 ^ (2 * (k - n) - 1)) = 0 := by
    rw [← pow_two]
    exact hd
  have e1 : (1 + denNat (k := k) (2 ^ (2 * (k - n) - 1))) * (1 - denNat (k := k) (2 ^ (2 * (k - n) - 1)))
      = 1 - denNat (k := k) (2 ^ (2 * (k - n) - 1)) * denNat (k := k) (2 ^ (2 * (k - n) - 1)) := by ring
  rw [e1, e2, sub_zero]

/-- **THE CONSTRUCTION (item 1, Class B).** For `0 < n < k` and `k ≤ 4·(k−n) − 2`, the
    scaled-halves pairing `scaledPair n (scaledSigma n)` with multiplier
    `scaledUnitary n = (2^(k−n), 2^(2(k−n)−1) − 1)` reproduces `rotr n` exactly **and** is
    norm one, hence a genuine rotation of the plane.

    The premise `k ≤ 4·(k−n) − 2` is *vacuous* for `n ≤ k/2 − 1/4` and covers every
    `n ≤ (3k−2)/4`; the uncovered range is the single amount `n = k−1` (§5). So the
    half-width boundary O-B is not a property of the amount. -/
theorem scaled_unitary_of_amount {k n : Nat} (hk : 1 ≤ k) (hn : 0 < n) (hnk : n < k)
    (h4 : k ≤ 4 * (k - n) - 2) :
    (∀ a : Nat, ellipticRead (scaledUnitary k n)
        (scaledPair (k := k) (n := n) (scaledSigma k n) a)
        = denNat (rotrVal (k := k) n a))
      ∧ ringNormForm (-1 : ZMod (2 ^ k)) (scaledUnitary k n) = 1 := by
  have hkn : 0 < k - n := by omega
  have h2m : k ≤ 2 * (2 * (k - n) - 1) := by omega
  have hD2 : (denNat (2 ^ (2 * (k - n) - 1)) : ZMod (2 ^ k)) ^ 2 = 0 := by
    simp only [denNat]
    rw [← Nat.cast_pow, ZMod.natCast_eq_zero_iff, ← pow_mul]
    rw [Nat.pow_dvd_pow_iff_le_right (show (1 : Nat) < 2 by norm_num)]
    omega
  have hsplit : (denNat (2 ^ (2 * (k - n) - 1)) : ZMod (2 ^ k))
      = 1 + denNat (2 ^ (2 * (k - n) - 1) - 1) := by
    have hnat : 2 ^ (2 * (k - n) - 1) = (2 ^ (2 * (k - n) - 1) - 1) + 1 := by
      have hpos : 1 ≤ 2 ^ (2 * (k - n) - 1) := Nat.one_le_two_pow
      omega
    have hcast : ((2 ^ (2 * (k - n) - 1) : Nat) : ZMod (2 ^ k))
        = (((2 ^ (2 * (k - n) - 1) - 1) + 1 : Nat) : ZMod (2 ^ k)) :=
      congrArg (fun m : Nat => (m : ZMod (2 ^ k))) hnat
    simp only [denNat]
    rw [hcast, Nat.cast_add, Nat.cast_one]
    ring
  have hBD : (denNat (2 ^ (2 * (k - n) - 1) - 1) : ZMod (2 ^ k))
      * (1 + denNat (2 ^ (2 * (k - n) - 1))) = -1 := by
    have hsub : (denNat (2 ^ (2 * (k - n) - 1) - 1) : ZMod (2 ^ k))
        = denNat (2 ^ (2 * (k - n) - 1)) - 1 := by
      rw [hsplit]
      ring
    have hcalc : (denNat (k := k) (2 ^ (2 * (k - n) - 1)) - 1)
        * (1 + denNat (k := k) (2 ^ (2 * (k - n) - 1)))
        = denNat (k := k) (2 ^ (2 * (k - n) - 1)) ^ 2 - 1 := by ring
    rw [hsub, hcalc, hD2]
    ring
  have hA2 : (denNat (2 ^ (k - n)) : ZMod (2 ^ k)) ^ 2
      = 2 * denNat (2 ^ (2 * (k - n) - 1)) := by
    have hnat : (2 ^ (k - n)) ^ 2 = 2 * 2 ^ (2 * (k - n) - 1) := by
      have h2 : (2 ^ (k - n)) ^ 2 = 2 ^ (2 * (k - n)) := by
        rw [← pow_mul]
        congr 1
        ring
      have h3 : 2 ^ (2 * (k - n)) = 2 ^ ((2 * (k - n) - 1) + 1) :=
        congrArg (fun e => 2 ^ e) (by omega)
      rw [h2, h3, pow_succ]
      ring
    simp only [denNat, ← Nat.cast_pow, hnat]
    push_cast
    ring
  have hs : scaledSigma k n = 1 + denNat (2 ^ (2 * (k - n) - 1)) := by
    simp only [scaledSigma, denNat, Nat.cast_add, Nat.cast_one]
  constructor
  · intro a
    simp only [scaledUnitary, scaledPair, halfPair_read_eq]
    have hcoef1 : (denNat (2 ^ (k - n)) : ZMod (2 ^ k)) * 1
        - denNat (2 ^ (2 * (k - n) - 1) - 1) * 0 = denNat (2 ^ (k - n)) := by ring
    have hA0 : (denNat (2 ^ (k - n)) : ZMod (2 ^ k)) * 0 = 0 := by ring
    have hcoef2 : (denNat (2 ^ (k - n)) : ZMod (2 ^ k)) * 0
        - denNat (2 ^ (2 * (k - n) - 1) - 1) * scaledSigma k n = 1 := by
      rw [hs, hA0, zero_sub, hBD]
      ring
    rw [hcoef1, hcoef2, one_mul]
    rw [denNat_rotrVal]
    ring
  · have hnd : (denNat (2 ^ (k - n)) : ZMod (2 ^ k)) ^ 2
        + (denNat (2 ^ (2 * (k - n) - 1) - 1)) ^ 2 = 1 := by
      have hsub : (denNat (2 ^ (2 * (k - n) - 1) - 1) : ZMod (2 ^ k))
          = denNat (2 ^ (2 * (k - n) - 1)) - 1 := by
        rw [hsplit]
        ring
      rw [hA2, hsub]
      have hexp : 2 * denNat (k := k) (2 ^ (2 * (k - n) - 1))
            + (denNat (k := k) (2 ^ (2 * (k - n) - 1)) - 1) ^ 2
          = 2 * denNat (k := k) (2 ^ (2 * (k - n) - 1))
            + (denNat (k := k) (2 ^ (2 * (k - n) - 1)) ^ 2
              - 2 * denNat (k := k) (2 ^ (2 * (k - n) - 1)) + 1) := by ring
      rw [hexp, hD2]
      ring
    simp only [ellipticNorm_eq, scaledUnitary]
    exact hnd

/-- **VERDICT (b).** The `n > k/2` amounts carry unit-circle multipliers under Class B
    pairings: at `k = 32` every amount `1 ≤ n ≤ 22` has one, and `17`, `18`, `19` lie
    strictly above `k/2`. AR-17's boundary is a property of the two named pairings, whose
    scale factors are both `1`, which pins the forced multiplier to `(2^(k−n), −1)`. -/
theorem verdict_unitary_beyond_half :
    (∀ n : ℕ, 1 ≤ n → n ≤ 22 →
        ∃ (σ : ZMod (2 ^ 32)) (u : ZMod (2 ^ 32) × ZMod (2 ^ 32)),
          ringNormForm (-1 : ZMod (2 ^ 32)) u = 1
            ∧ (∀ a : Nat, ellipticRead u (scaledPair (k := 32) (n := n) σ a)
                = denNat (rotrVal (k := 32) n a)))
      ∧ ¬ ((17 : ℕ) ≤ 32 / 2) ∧ ¬ ((18 : ℕ) ≤ 32 / 2) ∧ ¬ ((19 : ℕ) ≤ 32 / 2) := by
  refine ⟨?_, by decide, by decide, by decide⟩
  intro n h1 h22
  obtain ⟨hrr, hnn⟩ := scaled_unitary_of_amount (k := 32) (n := n) (by omega) (by omega)
    (by omega) (by omega)
  exact ⟨scaledSigma 32 n, scaledUnitary 32 n, hnn, hrr⟩

/-- **SHA-256's four rotation amounts are all elliptic-unitary under a Class B pairing at
    `k = 32`** — including the three AR-17 found off the circle. -/
theorem sha256_amounts_unitary : ∀ n ∈ ({7, 17, 18, 19} : Finset ℕ),
    (∀ a : Nat, ellipticRead (scaledUnitary 32 n)
        (scaledPair (k := 32) (n := n) (scaledSigma 32 n) a) = denNat (rotrVal (k := 32) n a))
      ∧ ringNormForm (-1 : ZMod (2 ^ 32)) (scaledUnitary 32 n) = 1 := by
  intro n hn
  have hn' : n = 7 ∨ n = 17 ∨ n = 18 ∨ n = 19 := by
    simpa only [Finset.mem_insert, Finset.mem_singleton, or_false, false_or] using hn
  rcases hn' with rfl | rfl | rfl | rfl
  · exact scaled_unitary_of_amount (k := 32) (n := 7) (by omega) (by omega) (by omega)
      (by omega)
  · exact scaled_unitary_of_amount (k := 32) (n := 17) (by omega) (by omega) (by omega)
      (by omega)
  · exact scaled_unitary_of_amount (k := 32) (n := 18) (by omega) (by omega) (by omega)
      (by omega)
  · exact scaled_unitary_of_amount (k := 32) (n := 19) (by omega) (by omega) (by omega)
      (by omega)

/-! ## §5 Ground checks at `k = 4` and `k = 8` -/

/-- **Width 4, the smallest width, reproduces AR-17's boundary**: at `n = 3 > k/2` the
    square condition fails, so no Class B unitary multiplier exists. `decide` over
    `ZMod 16`. -/
theorem no_unitary_amount3_width4 :
    ¬ ∃ y : ZMod 16, y * y = 1 - denNat (k := 4) (2 ^ (4 - 3)) * denNat (k := 4) (2 ^ (4 - 3)) := by
  decide

/-- **Width 8 is the smallest width where a Class B unitary multiplier exists above
    `k/2`**: at `n = 5 > k/2 = 4` the square condition holds, with root `31`
    (`31² ≡ 193 ≡ 1 − 64 (mod 256)`). `decide` over `ZMod 256`. -/
theorem unitary_amount5_width8 :
    ∃ y : ZMod 256, y * y = 1 - denNat (k := 8) (2 ^ (8 - 5)) * denNat (k := 8) (2 ^ (8 - 5)) := by
  decide

/-- The root is `31`: `31 * 31 = 961 ≡ 193 (mod 256) = 1 − 64`. -/
theorem unitary_amount5_root_width8 :
    (31 : ZMod 256) * 31 = 1 - denNat (k := 8) (2 ^ (8 - 5)) * denNat (k := 8) (2 ^ (8 - 5)) := by
  decide

/-- **The reflection amount `n = k−1` is where the square condition fails**: at `k = 8`,
    `1 − 2^2 ≡ 253 (mod 256) ≡ 5 (mod 8)` is not a square. `decide` over `ZMod 256`. The
    general statement `reflecting_amount_no_square` (`¬ ∃ y : ZMod (2^k), y*y = 1 − 2^2` for
    `3 ≤ k`) is the named open lemma; see `AlgebraRotorUnit.md`. -/
theorem width8_quarter_value :
    (((1 - denNat (k := 8) (2 ^ (8 - 7)) * denNat (k := 8) (2 ^ (8 - 7)) : ZMod (2 ^ 8))).val) % 8 = 5 := by
  decide

/-- **The width-8 witness, on all `2^8` words**: multiplier `(8, 31)` through the pairing
    `halfPair 1 0 0 33` (i.e. `scaledPair 5 33`, since `2^(8−5) = 8` and
    `31 = −33⁻¹ (mod 256)`) reproduces `rotr 5`. -/
def width8Multiplier : ZMod (2 ^ 8) × ZMod (2 ^ 8) :=
  (denNat (k := 8) (2 ^ (8 - 5)), denNat (k := 8) 31)

def width8Pairing (a : Nat) : ZMod (2 ^ 8) × ZMod (2 ^ 8) :=
  halfPair (k := 8) (n := 5) 1 0 0 (denNat (k := 8) 33) a

theorem scaled_read_width8 :
    ∃ a : ZMod 256, ellipticRead width8Multiplier (width8Pairing a.val)
      = denNat (k := 8) (rotrVal (k := 8) 5 a.val) := by
  decide

theorem scaled_norm_width8 : ringNormForm (-1 : ZMod (2 ^ 8)) width8Multiplier = 1 := by decide

/-- AR-17's boundary at width 8: the unscaled bit-halves pairing forces `(8, −1)`, whose
    norm is `8² + 1 = 65 ≠ 1 (mod 256)`. -/
theorem halves_multiplier_off_circle_width8 :
    ringNormForm (-1 : ZMod 256) (denNat (k := 8) (2 ^ (8 - 5)), -1) ≠ 1 := by decide

/-- **AR-17's forced multiplier is the singleton of the generalised method at
    `(α,β,γ,δ) = (1,0,0,1)`.** -/
theorem halvesMultiplier_is_forced {k n : Nat} (hn : 0 < n) (c s : ZMod (2 ^ k))
    (hsol : c * 1 - s * 0 = denNat (2 ^ (k - n)) ∧ c * 0 - s * 1 = 1) :
    c = denNat (2 ^ (k - n)) ∧ s = -1 := by
  obtain ⟨h1, h2⟩ := hsol
  constructor
  · calc c = c * 1 - s * 0 := by ring
    _ = denNat (2 ^ (k - n)) := h1
  · calc s = -(c * 0 - s * 1) := by ring
    _ = -1 := congrArg Neg.neg h2

/-- **The family, exhibited: a singular Class B pairing at width 2 has a genuinely larger
    family of reproducing multipliers.** For `(α,β,γ,δ) = (2,1,0,0)` at `k = 2, n = 1` the
    system is `2c = 2`, `c = 1`, so every `s ∈ ZMod 4` is a multiplier. -/
theorem singular_family_width2 :
    (∀ (c s : ZMod 4), c * 2 - s * 0 = 2 → c * 1 - s * 0 = 1 → c = 1)
      ∧ ((1 : ZMod 4) * 2 - (0 : ZMod 4) * 0 = 2 ∧ (1 : ZMod 4) * 1 - (0 : ZMod 4) * 0 = 1)
      ∧ ((1 : ZMod 4) * 2 - (1 : ZMod 4) * 0 = 2 ∧ (1 : ZMod 4) * 1 - (1 : ZMod 4) * 0 = 1)
      ∧ ((0 : ZMod 4) * 0 - 2 * 0 = 0) := by decide

/-- That singular pairing does reproduce `rotr 1` at width 2, with `(1, 0)` as multiplier. -/
theorem singular_pairing_reproduces_width2 :
    ¬ ∃ x : ZMod 4, ellipticRead (denNat (k := 2) 1, denNat (k := 2) 0)
        (halfPair (k := 2) (n := 1) (denNat (k := 2) 2) (denNat (k := 2) 1)
          (denNat (k := 2) 0) (denNat (k := 2) 0) x.val)
        ≠ denNat (k := 2) (rotrVal (k := 2) 1 x.val) := by
  decide

/-! ## §6 The cos/sin carrier: `Nous.Model.RopeOrbit` -/

noncomputable section

open Nous.Model

/-- **The cos/sin pairing.** The word of value `a` is placed on the unit circle
    `T = {(c,s) | c²+s² = 1}` at the angle `2π·a/2^k`; this is a non-re-encoding carrier:
    it is not a function of two bit-windows of `a` at all. -/
def ropeCarrier (k a : Nat) : ℝ × ℝ := rho (2 * Real.pi * (a : ℝ) / 2 ^ k)

theorem ropeCarrier_zero (k : Nat) : ropeCarrier k 0 = (1, 0) := by
  simp [ropeCarrier, rho, Real.cos_zero, Real.sin_zero]

theorem ropeCarrier_cos (k a : Nat) :
    (ropeCarrier k a).1 = Real.cos (2 * Real.pi * (a : ℝ) / 2 ^ k) := rfl

/-- `rhoMul z (1, 0) = z`: the word `0` sits at the identity of the circle. -/
theorem rhoMul_one (z : ℝ × ℝ) : rhoMul z (1, 0) = z := by
  simp only [rhoMul]
  exact Prod.ext_iff.mpr ⟨by ring, by ring⟩

/-- **The carrier pins the multiplier at `a = 0`.** Because `ropeCarrier k 0 = (1, 0)` is the
    identity of `T`, any multiplier reproducing any value at the word `0` must be `(1, 0)`. -/
theorem rope_at_zero_forces_identity {k : Nat} {u : ℝ × ℝ}
    (h : rhoMul (ropeCarrier k 0) u = ropeCarrier k 0) : u = (1, 0) := by
  have hz := h
  rw [ropeCarrier_zero] at hz
  have h' : rhoMul (1, 0) u = u := by
    simp only [rhoMul]
    exact Prod.ext_iff.mpr ⟨by ring, by ring⟩
  rw [h'] at hz
  exact hz

/-- **Named obstruction O-E: the cos/sin carrier reproduces no nonzero rotation at all.**
    For every width `k ≥ 1` and amount `0 < n < k`, no multiplier — unitary or not —
    reproduces `rotr n` through `ropeCarrier`.

    At `a = 0` the multiplier is pinned to `(1, 0)` (`rope_at_zero_forces_identity`), so the
    carrier must map each word to itself; but `rotr n 1 = 2^(k−n) ≠ 1` and the two angles lie
    in `[0, π]` with the first strictly smaller, so injectivity of `cos` on `[0, π]`
    (`Real.injOn_cos`) refutes it. The cos/sin carrier is therefore a *stronger* obstruction
    than O-B, not a way around it. -/
theorem rope_no_rotr_mult {k n : Nat} (hk : 1 ≤ k) (hn : 0 < n) (hnk : n < k) :
    ¬ ∃ (u : ℝ × ℝ), u.1 ^ 2 + u.2 ^ 2 = 1 ∧
        ∀ a : Nat, a < 2 ^ k →
          rhoMul (ropeCarrier k a) u = ropeCarrier k (rotrVal (k := k) n a) := by
  rintro ⟨u, hun, h⟩
  have hzero := h 0 (Nat.two_pow_pos k)
  have hu : u = (1, 0) := by
    apply rope_at_zero_forces_identity
    have hz : rotrVal (k := k) n 0 = 0 := by simp [rotrVal]
    rw [hz] at hzero
    exact hzero
  have hone := h 1 (Nat.one_lt_two_pow (by omega : k ≠ 0))
  rw [hu, rhoMul_one, rotrVal_one' hn] at hone
  have hcos : Real.cos (2 * Real.pi * (1 : ℝ) / 2 ^ k)
      = Real.cos (2 * Real.pi * ((2 ^ (k - n) : Nat) : ℝ) / 2 ^ k) := by
    have hfst := congrArg Prod.fst hone
    unfold ropeCarrier at hfst
    simpa only [Nous.Model.rho, Prod.fst, Nat.cast_one] using hfst
  have hkR : (0 : ℝ) < (2 : ℝ) ^ k := by positivity
  have hmem1 : 2 * Real.pi * (1 : ℝ) / 2 ^ k ∈ Set.Icc (0 : ℝ) Real.pi := by
    constructor
    · positivity
    · rw [div_le_iff₀ hkR]
      have hp2 : (2 : ℝ) ≤ (2 : ℝ) ^ k := by
        exact_mod_cast Nat.succ_le_of_lt (Nat.one_lt_two_pow (by omega : k ≠ 0))
      have h := mul_le_mul_of_nonneg_left hp2 Real.pi_pos.le
      linarith
  have hmem2 : 2 * Real.pi * ((2 ^ (k - n) : Nat) : ℝ) / 2 ^ k ∈ Set.Icc (0 : ℝ) Real.pi := by
    constructor
    · positivity
    · rw [div_le_iff₀ hkR]
      have hp3 : (2 : ℝ) * ((2 ^ (k - n) : Nat) : ℝ) ≤ (2 ^ k : ℝ) := by
        have hnat : 2 * 2 ^ (k - n) ≤ 2 ^ k := by
          have h1 : 2 * 2 ^ (k - n) = 2 ^ ((k - n) + 1) := by
            rw [pow_succ]
            ring
          rw [h1]
          exact Nat.pow_le_pow_right (by norm_num) (by omega)
        exact_mod_cast hnat
      have h := mul_le_mul_of_nonneg_left hp3 Real.pi_pos.le
      linarith
  have hne : 2 * Real.pi * (1 : ℝ) / 2 ^ k
      ≠ 2 * Real.pi * ((2 ^ (k - n) : Nat) : ℝ) / 2 ^ k := by
    have hX : (1 : ℝ) < ((2 ^ (k - n) : Nat) : ℝ) := by
      exact_mod_cast Nat.one_lt_two_pow (by omega : k - n ≠ 0)
    have hlt : 2 * Real.pi * (1 : ℝ) / 2 ^ k
        < 2 * Real.pi * ((2 ^ (k - n) : Nat) : ℝ) / 2 ^ k := by
      rw [div_lt_div_iff_of_pos_right hkR]
      exact mul_lt_mul_of_pos_left hX (by positivity : (0 : ℝ) < 2 * Real.pi)
    exact ne_of_lt hlt
  exact hne (Real.injOn_cos hmem1 hmem2 hcos)

/-! ## §7 Axiom cone -/

#print axioms ringMul_cell_eq
#print axioms ellipticRead_eq
#print axioms ellipticNorm_eq
#print axioms rotrVal_eq_rotrBv
#print axioms denNat_rotrVal
#print axioms halfPair_read_eq
#print axioms halfPair_at_one
#print axioms halfPair_at_pow
#print axioms rotrVal_one'
#print axioms rotrVal_pow'
#print axioms halfPair_reproduces_iff
#print axioms halfPair_unique
#print axioms halfPair_family_add
#print axioms affine_read_eq
#print axioms affine_pairing_no_rotr
#print axioms scaledPair_lo
#print axioms scaled_read_rotr
#print axioms scaled_forced
#print axioms scaled_unitary_iff_square
#print axioms scaled_sigma_inv
#print axioms scaled_unitary_of_amount
#print axioms verdict_unitary_beyond_half
#print axioms sha256_amounts_unitary
#print axioms no_unitary_amount3_width4
#print axioms unitary_amount5_width8
#print axioms unitary_amount5_root_width8
#print axioms width8_quarter_value
#print axioms scaled_read_width8
#print axioms width8Multiplier
#print axioms width8Pairing
#print axioms scaled_norm_width8
#print axioms halves_multiplier_off_circle_width8
#print axioms halvesMultiplier_is_forced
#print axioms singular_family_width2
#print axioms singular_pairing_reproduces_width2
#print axioms ropeCarrier_zero
#print axioms ropeCarrier_cos
#print axioms rhoMul_one
#print axioms rope_at_zero_forces_identity
#print axioms rope_no_rotr_mult

end

end AlgebraRotorUnit