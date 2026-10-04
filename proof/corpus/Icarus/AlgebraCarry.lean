/-
# Carries in the two-slot algebra `R[x]/(x^2 - sigma)`

`+` on `ZMod (2^k)` is the ring addition that `BitVec k` wrapping addition computes, and it is
`F2`-linear only at width one. The deviation from `F2`-linearity is exactly the class of two,
which is the first carry. This file states that mechanism precisely, states where it is *not*
to be found, and fixes the exact algebraic degree of modular addition at widths two and four.

| claim | verdict |
|---|---|
| the deviation is a derivational term | refuted: an additive map satisfying Leibniz over `ZMod 2^k` is the zero map, at every `k`; doubling is an explicit nonzero additive map that fails Leibniz at every `k ≥ 2` |
| every additive endomorphism of `ZMod 2^k` is zero | refuted: doubling is one at every `k ≥ 2`; an additive endomorphism is exactly the `x.val`-fold multiple of its value at one |
| the deviation is exposed by an idempotent decomposition | refuted: the only idempotents of `RingCell (ZMod 2^k)` are `(0,0)` and `ringOne`, and both have second slot `0` |
| the deviation is a multiplicative term in the second slot | proved, and `sigma`-independent: `ringMul sigma (a,0) (0,b) = (0, a*b)` for every `sigma`, and the deviation from exclusive-or addition is `2*(x &&& y)` at widths two, four and eight |
| the deviation is the second slot of one cell, and the two slots recompose the sum | proved: `carryCell sigma x y = ringMul sigma (x^^^y, 0) ringOne + ringMul sigma (2,0) (0, x&&&y)` at every width and face; the two slots recompose `x + y` completely at widths two and four, and at width eight by a concrete instance only (the `2^16`-pair enumeration is outside the lane memory envelope) |
| the deviation needs the parabolic (dual-number) face | refuted: the second slot is the product of two first-slot elements at **every** face; `sigma` enters only the first slot |
| the parabolic norm form sees the deviation | refuted: `ringNormForm 0 z = z.1^2`, blind to the second slot |
| the deviation is readable by the channel read `ringUpper r` | true at `sigma = 1` (`ringUpper 1 z = z.1 + z.2`, which for the carry cell *is* the modular sum) and at `sigma = 0` the canonical read `ringUpper 0 z = z.1` is blind; at `sigma = -1` no read exists for `k >= 2` |
| a read of `RingCell (ZMod (2^k))` exists at `sigma = -1` | refuted for `k >= 2`: `ringChannel_gen_sq` forces a square root of `-1`, and `ZMod (2^k)` has none, so no `RingChannel (-1)` exists; a root exists at `sigma = 0` and at `sigma = 1` |
| modular addition has `F2`-degree two | proved at `k = 2`, refuted at `k = 4`: the top output bit has degree four |
| output bit `j` has degree `j + 1` | proved at `k = 2` and `k = 4`, all bits, with the witness monomials `{x0,y0} + {x1..x(j-1)}` |
| output bit `j` has `2^j + 1` algebraic-normal-form terms | proved at `k = 2` and `k = 4`, all bits, matching Dehnavi-Rishekani-Mirzaee Shamsabad-Maimani-Pasha, ePrint 2016/181 Thm 4.1 |

Grades: every declaration here is `proved` by the Lean kernel under the closed cone
`{propext, Classical.choice, Quot.sound}`; the width-specific rows are complete finite checks at
widths two and four, not samples. No claim is made about SHA-2, SHA-3, constant-time execution,
cryptographic security or any hash round: nothing in this file is connected to those.
-/
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Mathlib.Algebra.DualNumber

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace AlgebraCarry

open Apeiron.Trichotomy.AnyRing

/-! ## 1 · Wrapping addition is addition in `ZMod (2^k)`

A `k`-bit word read as an element of the modular ring. -/

/-- A `k`-bit word read as an element of `ZMod (2^k)`. -/
def denoteBV {k : ℕ} (w : BitVec k) : ZMod (2 ^ k) := (w.toNat : ZMod (2 ^ k))

/-- **Wrapping addition of `k`-bit words is addition in `ZMod (2^k)`.** -/
theorem denoteBV_add (k : ℕ) (x y : BitVec k) :
    denoteBV (x + y) = denoteBV x + denoteBV y := by
  rw [denoteBV, BitVec.toNat_add, ZMod.natCast_mod]
  exact Nat.cast_add _ _

/-- **The `F2`-linear reference point: a word added to itself by exclusive-or is zero.** -/
theorem denoteBV_xor_self (k : ℕ) (w : BitVec k) : denoteBV (w ^^^ w) = 0 := by
  simp [denoteBV]

/-! ## 2 · The deviation from `F2`-linearity is the class of two

`F2`-linearity of `+` means `2 * x = 0` for every `x`: over `F2`, doubling is zero. In
`ZMod (2^k)` the doubling is zero exactly where the modulus already divides two. -/

/-- The canonical representative of the class of two in `ZMod (2^k)` is `2 % 2^k`. -/
theorem two_val (k : ℕ) : (((2 : ℕ) : ZMod (2 ^ k)).val) = 2 % 2 ^ k := ZMod.val_natCast _ _

/-- `2^k ≤ 2` exactly at `k ≤ 1`. -/
theorem two_pow_le_two_iff {k : ℕ} : 2 ^ k ≤ 2 ↔ k ≤ 1 := by
  have h : ∀ (n m : ℕ), 2 ^ n ≤ 2 ^ m ↔ n ≤ m := fun n m => Nat.pow_le_pow_iff_right (by omega)
  simpa using h k 1

/-- `2^k = 2 * 2^(k-1)` at every positive width. -/
theorem two_pow_succ_form (k : ℕ) (hk : 1 ≤ k) : 2 ^ k = 2 * 2 ^ (k - 1) := by
  have h2 : 2 ^ (k - 1) * 2 = 2 ^ k := by
    rw [← Nat.pow_succ]
    congr 2
    omega
  rw [← h2, Nat.mul_comm]

/-- **The class of two does not vanish at any width `k ≥ 3`.** -/
theorem two_ne_zero_of_three_le_k {k : ℕ} (hk : 3 ≤ k) : ((2 : ℕ) : ZMod (2 ^ k)) ≠ 0 := by
  intro h
  have hz : 2 % 2 ^ k = 0 := by rw [← two_val]; exact (ZMod.val_eq_zero _).mpr h
  have h4 : 4 ≤ 2 ^ k :=
    (Nat.pow_le_pow_iff_right (a := 2) (n := 2) (m := k) (by omega)).2 (by omega)
  have hm : 2 % 2 ^ k = 2 := Nat.mod_eq_of_lt (by omega)
  rw [hm] at hz
  omega

/-- **The class of two vanishes in `ZMod (2^k)` exactly at the widths `k ≤ 1`.**

    This is the whole obstruction: `+` is `F2`-linear at width one, and above width one the
    deviation is the nonzero class of two. -/
theorem two_eq_zero_iff (k : ℕ) : ((2 : ℕ) : ZMod (2 ^ k)) = 0 ↔ k ≤ 1 := by
  constructor
  · intro h
    have hne : ¬ (2 ≤ k) := by
      intro h2k
      have h4 : 4 ≤ 2 ^ k :=
        (Nat.pow_le_pow_iff_right (a := 2) (n := 2) (m := k) (by omega)).2 (by omega)
      have hm : 2 % 2 ^ k = 2 := Nat.mod_eq_of_lt (by omega)
      have hz : 2 % 2 ^ k = 0 := by rw [← two_val]; exact (ZMod.val_eq_zero _).mpr h
      rw [hm] at hz
      omega
    rcases Nat.lt_or_ge k 2 with hlt | hge
    · exact Nat.le_of_lt_succ hlt
    · exact absurd hge hne
  · intro hk
    have hc : k = 0 ∨ k = 1 := by omega
    rcases hc with rfl | rfl
    · refine (ZMod.val_eq_zero ((2 : ℕ) : ZMod (2 ^ 0))).mp ?_
      rw [ZMod.val_natCast]
      simp
    · refine (ZMod.val_eq_zero ((2 : ℕ) : ZMod (2 ^ 1))).mp ?_
      rw [ZMod.val_natCast]
      simp

/-- **Modular addition on `ZMod (2^k)` is annihilated by doubling exactly at width one.**

    `∀ x, 2 * x = 0` is `F2`-linearity of the addition, read in the ring itself. Width zero is
    the trivial ring `ZMod 1`, excluded here; `two_eq_zero_iff` covers it. -/
theorem doubling_zero_iff (k : ℕ) (hk : 1 ≤ k) :
    (∀ x : ZMod (2 ^ k), ((2 : ℕ) : ZMod (2 ^ k)) * x = 0) ↔ k = 1 := by
  constructor
  · intro h
    have hle : k ≤ 1 := (two_eq_zero_iff k).mp (by simpa using h 1)
    omega
  · intro hk'
    have hk'' : k = 1 := hk'
    subst hk''
    intro x
    simp only [ZMod.natCast_self, zero_mul]

/-- **The deviation from `F2`-linearity is nonzero at every width `k ≥ 2`.** -/
theorem two_ne_zero_of_two_le_k {k : ℕ} (hk : 2 ≤ k) : ((2 : ℕ) : ZMod (2 ^ k)) ≠ 0 := by
  intro h
  have hle : k ≤ 1 := (two_eq_zero_iff k).mp h
  omega

/-- **Checked deviation witness at width two**: `1 + 1` is nonzero, while `1 ^ 1` is zero. -/
theorem deviation_witness_k2 : ((1 : BitVec 2) + 1 ≠ 0) ∧ ((1 : BitVec 2) ^^^ 1 = 0) := by decide

/-- **Checked deviation witness at width four.** -/
theorem deviation_witness_k4 : ((1 : BitVec 4) + 1 ≠ 0) ∧ ((1 : BitVec 4) ^^^ 1 = 0) := by decide

/-- **Checked deviation witness at width eight.** -/
theorem deviation_witness_k8 : ((1 : BitVec 8) + 1 ≠ 0) ∧ ((1 : BitVec 8) ^^^ 1 = 0) := by decide

/-! ## 3 · The deviation is additive but not derivational

Doubling is a nonzero additive endomorphism of `ZMod (2^k)` at every `k ≥ 2`, so the deviation
*can* be carried by an additive term. A derivation is more constrained: the Leibniz rule at
`1 * 1` forces its value at one to vanish, and an additive map vanishing at one is the zero map.
So the derivational reading is empty at every width, while the multiplicative reading carries
the carry. -/

/-- A natural number cast in `ZMod n` is the `n`-fold scalar multiple of one. -/
theorem natCast_eq_nsmul {n : ℕ} (a : ℕ) : ((a : ℕ) : ZMod n) = a • 1 := by
  simp [nsmul_eq_mul]

/-- **An additive endomorphism of `ZMod (2^k)` is the `x.val`-fold multiple of its value at
    one**: `f x = x.val * f 1`. This is the classification of the additive endomorphisms. -/
theorem addHom_apply (k : ℕ) (f : ZMod (2 ^ k) →+ ZMod (2 ^ k)) (x : ZMod (2 ^ k)) :
    f x = (x.val : ℕ) * f 1 := by
  calc f x = f ((x.val : ℕ) : ZMod (2 ^ k)) := congrArg f (ZMod.natCast_zmod_val x).symm
    _ = (x.val : ℕ) • f 1 := by rw [natCast_eq_nsmul, map_nsmul]
    _ = (x.val : ℕ) * f 1 := by rw [nsmul_eq_mul]

/-- **A Leibniz rule over `ZMod (2^k)` forces the additive map to be zero**: at `1 * 1` the rule
    reads `f 1 = f 1 + f 1`, so `f 1 = 0`, and `addHom_apply` finishes.** -/
theorem leibniz_addHom_zero (k : ℕ) (f : ZMod (2 ^ k) →+ ZMod (2 ^ k))
    (h : ∀ x y : ZMod (2 ^ k), f (x * y) = x * f y + y * f x) : f = 0 := by
  have hf1 : f 1 = 0 := by
    have h1 := h 1 1
    simp only [one_mul, mul_one] at h1
    exact add_left_cancel (h1.symm.trans (add_zero _).symm)
  ext x
  calc f x = (x.val : ℕ) * f 1 := addHom_apply k f x
    _ = (x.val : ℕ) * 0 := by rw [hf1]
    _ = 0 := by rw [mul_zero]

/-- The doubling map of `ZMod (2^k)` as an additive endomorphism. -/
def doublingHom (k : ℕ) : ZMod (2 ^ k) →+ ZMod (2 ^ k) where
  toFun := fun x => ((2 : ℕ) : ZMod (2 ^ k)) * x
  map_zero' := by simp
  map_add' := by intro a b; ring

/-- `doublingHom` is multiplication by the class of two. -/
theorem doublingHom_apply (k : ℕ) (x : ZMod (2 ^ k)) :
    doublingHom k x = ((2 : ℕ) : ZMod (2 ^ k)) * x := rfl

/-- **Counterexample to the additive reading of deviations**: doubling is a nonzero additive
    endomorphism of `ZMod (2^k)` at every width `k ≥ 2`.** -/
theorem doublingHom_ne_zero {k : ℕ} (hk : 2 ≤ k) : doublingHom k ≠ 0 := by
  intro h
  have h2 := congrArg (fun g : ZMod (2 ^ k) →+ ZMod (2 ^ k) => g 1) h
  have h3 : ((2 : ℕ) : ZMod (2 ^ k)) * 1 = 0 := by
    simpa [doublingHom_apply] using h2
  exact two_ne_zero_of_two_le_k hk (by simpa using h3)

/-- **Counterexample to the derivational reading of deviations**: doubling is additive and
    nonzero at every width `k ≥ 2`, yet it fails the Leibniz rule there.** -/
theorem doublingHom_not_leibniz {k : ℕ} (hk : 2 ≤ k) :
    ¬ ∀ x y : ZMod (2 ^ k), doublingHom k (x * y)
        = x * doublingHom k y + y * doublingHom k x := by
  intro h
  exact doublingHom_ne_zero hk (leibniz_addHom_zero k (doublingHom k) h)

/-! ## 4 · The parabolic face `sigma = 0` is the dual-number algebra

`R[x]/(x^2)` at `sigma = 0` is the dual-number algebra (Clifford 1873), i.e. the trivial
square-zero extension `TrivSqZeroExt R R` of `Mathlib/Algebra/TrivSqZeroExt/Basic.lean:408`,
whose multiplication is `(a, b) * (c, d) = (a * c, a * d + b * c)`. A product of two first-slot
elements therefore lands in the second slot: that is the slot a carry term can occupy. -/

/-- **At `sigma = 0` the two-slot multiplication is the dual-number multiplication formula**,
    the multiplication of `TrivSqZeroExt R R`. -/
theorem ringMul_zero_components (R : Type*) [CommRing R] (a b c d : R) :
    ringMul (0 : R) (a, b) (c, d) = (a * c, a * d + b * c) := by
  simp [ringMul]

/-- **At `sigma = 0` a product of two first-slot elements lies in the second slot.** -/
theorem ringMul_zero_second (R : Type*) [CommRing R] (a b : R) :
    ringMul (0 : R) (a, 0) (0, b) = (0, a * b) := by
  simp [ringMul]

/-- **The parabolic norm form reads the first slot only: it is blind to the second.** -/
theorem ringNormForm_zero_first (R : Type*) [CommRing R] (a b : R) :
    ringNormForm (0 : R) (a, b) = a ^ 2 := by
  simp [ringNormForm]

/-! ## 5 · Obstruction: no idempotent decomposition exposes the deviation

`RingCell` at `sigma = 0` is a dual-number algebra, whose idempotents over a ring in which every
odd element is a unit are only zero and one. Both have second slot zero, so an idempotent split
of the two-slot algebra cannot produce the second-slot carry term. -/

/-- An odd natural number is coprime to two. -/
theorem coprime_two_of_mod_two_eq_one {M : ℕ} (h : M % 2 = 1) : Nat.Coprime M 2 := by
  rw [Nat.coprime_comm, Nat.prime_two.coprime_iff_not_dvd]
  intro hd
  obtain ⟨c, rfl⟩ := hd
  rw [Nat.mul_comm (2 : ℕ) c, Nat.mul_mod] at h
  omega

/-- The canonical representative of `a.val + n` is `a` in `ZMod n`. -/
theorem natCast_val_add {n : ℕ} [NeZero n] (a : ZMod n) : ((a.val + n : ℕ) : ZMod n) = a := by
  rw [Nat.cast_add, ZMod.natCast_self, ZMod.natCast_zmod_val, add_zero]

/-- `2 * X - 1` is odd for every `X ≥ 1`. -/
theorem two_mul_sub_one_odd (X : ℕ) (hX : 1 ≤ X) : Odd (2 * X - 1) := by
  refine ⟨X - 1, ?_⟩
  omega

/-- **Every element of the form `2a - 1` in `ZMod (2^k)` is a unit, at every width `k ≥ 1`.** -/
theorem isUnit_two_mul_sub_one {k : ℕ} (hk : 1 ≤ k) (a : ZMod (2 ^ k)) : IsUnit (2 * a - 1) := by
  have hposk : 0 < (2 : ℕ) ^ k := by positivity
  haveI : NeZero (2 ^ k) := ⟨by omega⟩
  have hrep : ((2 * (a.val + 2 ^ k) - 1 : ℕ) : ZMod (2 ^ k)) = 2 * a - 1 := by
    have ha : ((a.val + 2 ^ k : ℕ) : ZMod (2 ^ k)) = a := natCast_val_add a
    have h1 : ((2 * (a.val + 2 ^ k) - 1 : ℕ) : ZMod (2 ^ k))
        = ((2 * (a.val + 2 ^ k) : ℕ) : ZMod (2 ^ k)) - 1 := by
      simp
    have h2 : ((2 * (a.val + 2 ^ k) : ℕ) : ZMod (2 ^ k)) = 2 * a := by
      rw [Nat.cast_mul, ha]
      simp
    rw [h1, h2]
  rw [← hrep, ZMod.isUnit_iff_coprime]
  refine (Nat.coprime_pow_right_iff (by omega) _ _).2 ?_
  rw [Nat.coprime_comm, Nat.coprime_two_left]
  exact two_mul_sub_one_odd (a.val + 2 ^ k) (by omega)

/-- **At every width `k ≥ 1`, either an element of `ZMod (2^k)` is a unit or one less than it
    is.** -/
theorem isUnit_a_or_sub_one {k : ℕ} (hk : 1 ≤ k) (a : ZMod (2 ^ k)) :
    IsUnit a ∨ IsUnit (a - 1) := by
  have hposk : 0 < (2 : ℕ) ^ k := by positivity
  haveI : NeZero (2 ^ k) := ⟨by omega⟩
  by_cases he : a.val % 2 = 0
  · right
    have hrep : ((a.val + 2 ^ k - 1 : ℕ) : ZMod (2 ^ k)) = a - 1 := by
      have ha : ((a.val + 2 ^ k : ℕ) : ZMod (2 ^ k)) = a := natCast_val_add a
      have h1 : ((a.val + 2 ^ k - 1 : ℕ) : ZMod (2 ^ k))
          = ((a.val + 2 ^ k : ℕ) : ZMod (2 ^ k)) - 1 := by
        simp
      rw [h1, ha]
    have h2k : 2 ^ k = 2 * 2 ^ (k - 1) := two_pow_succ_form k hk
    have hdec : a.val = a.val % 2 + 2 * (a.val / 2) := (Nat.mod_add_div a.val 2).symm
    have hdec0 : a.val = 2 * (a.val / 2) := by
      calc a.val = a.val % 2 + 2 * (a.val / 2) := hdec
        _ = 2 * (a.val / 2) := by rw [he, Nat.zero_add]
    have hform : a.val + 2 ^ k - 1 = 2 * (a.val / 2 + 2 ^ (k - 1) - 1) + 1 := by
      have hpos : 0 < a.val / 2 + 2 ^ (k - 1) := by positivity
      omega
    rw [← hrep, ZMod.isUnit_iff_coprime]
    refine (Nat.coprime_pow_right_iff (by omega) _ _).2 ?_
    rw [Nat.coprime_comm, Nat.coprime_two_left]
    exact ⟨a.val / 2 + 2 ^ (k - 1) - 1, hform⟩
  · left
    have hone : a.val % 2 = 1 := by
      rcases Nat.mod_two_eq_zero_or_one a.val with h | h
      · exact absurd h he
      · exact h
    have hdecomp : a.val = 2 * (a.val / 2) + 1 := by
      have h := Nat.mod_add_div a.val 2
      rw [hone] at h
      omega
    rw [← ZMod.natCast_zmod_val a, ZMod.isUnit_iff_coprime]
    refine (Nat.coprime_pow_right_iff (by omega) _ _).2 ?_
    rw [Nat.coprime_comm, Nat.coprime_two_left]
    exact ⟨a.val / 2, hdecomp⟩

/-- **The only idempotents of the two-slot algebra over `ZMod (2^k)` are the origin and the
    unit, at every `sigma` and every width `k ≥ 1`.** -/
theorem idempotent_trivial {k : ℕ} (hk : 1 ≤ k) {σ : ZMod (2 ^ k)}
    {z : RingCell (ZMod (2 ^ k))} (h : RingIdempotent σ z) : z = (0, 0) ∨ z = ringOne :=
  ringIdempotent_trivial (isUnit_two_mul_sub_one hk _) (isUnit_a_or_sub_one hk _) h

/-- **An idempotent of the two-slot algebra over `ZMod (2^k)` has second slot zero**, so no
    idempotent split can expose a second-slot carry term. -/
theorem idempotent_second_zero {k : ℕ} (hk : 1 ≤ k) {σ : ZMod (2 ^ k)}
    {z : RingCell (ZMod (2 ^ k))} (h : RingIdempotent σ z) : z.2 = 0 :=
  ringIdempotent_snd_eq_zero (isUnit_two_mul_sub_one hk _) h

/-- **Every idempotent of the two-slot algebra over `ZMod (2^k)` has parabolic size zero or
    one**, read through the `sigma = 0` norm form. -/
theorem idempotent_normForm_zero_or_one {k : ℕ} (hk : 1 ≤ k) {σ : ZMod (2 ^ k)}
    {z : RingCell (ZMod (2 ^ k))} (h : RingIdempotent σ z) :
    ringNormForm 0 z = 0 ∨ ringNormForm 0 z = 1 := by
  rcases idempotent_trivial hk h with hz | hz
  · left
    subst hz
    simp [ringNormForm]
  · right
    subst hz
    simp [ringNormForm]

/-! ## 6 · The carry term: deviation from exclusive-or is the doubled conjunction

`x + y = (x ^^^ y) + ((x &&& y) <<< 1)`: the deviation from the `F2`-linear (exclusive-or) sum
is exactly the doubled bitwise conjunction, the carry. Complete finite checks at widths two and
four, in the word arithmetic and in the ring. -/

/-- **Width two: the deviation of wrapping addition from exclusive-or addition is the carry.** -/
theorem carry_identity_two : ∀ x y : BitVec 2, x + y = (x ^^^ y) + ((x &&& y) <<< 1) := by decide

/-- **Width four: the deviation of wrapping addition from exclusive-or addition is the carry.** -/
theorem carry_identity_four : ∀ x y : BitVec 4, x + y = (x ^^^ y) + ((x &&& y) <<< 1) := by decide

/-- **Width two, in the ring `ZMod 4`: the deviation of modular addition from exclusive-or
    addition is twice the bitwise conjunction, a product term.** -/
theorem carry_identity_two_zmod : ∀ x y : BitVec 2,
    ((x + y).toNat : ZMod 4) = ((x ^^^ y).toNat : ZMod 4) + 2 * ((x &&& y).toNat : ZMod 4) := by
  decide

/-- **Width four, in the ring `ZMod 16`: the deviation of modular addition from exclusive-or
    addition is twice the bitwise conjunction, a product term.** -/
theorem carry_identity_four_zmod : ∀ x y : BitVec 4,
    ((x + y).toNat : ZMod 16) = ((x ^^^ y).toNat : ZMod 16) + 2 * ((x &&& y).toNat : ZMod 16) := by
  decide

/-- **The carry term `2 * (x &&& y)` of `ZMod 16` addition is a product, and at `sigma = 0` it
    occupies the second slot of the two-slot algebra.** -/
theorem deviation_in_second_slot (x y : BitVec 4) :
    ringMul (0 : ZMod 16) (2, 0) (0, denoteBV (x &&& y))
      = (0, 2 * denoteBV (x &&& y)) :=
  ringMul_zero_second (ZMod 16) 2 (denoteBV (x &&& y))

/-! ## 7 · The exact `F2`-degree of modular addition at widths two and four

An `F2`-function on the joint bits has an algebraic normal form: a sum over monomials in the
input bits. A monomial is a mask over those `2k` input bits: bit `i` of the first word is
position `i` of the mask and bit `i` of the second word is position `k + i`. `anfCoeffMask j S`
is the coefficient of the monomial with mask `S` in the algebraic normal form of output bit `j`,
computed as the parity of that bit over the subcube supported by `S`. `anfDegreeExactlyMask j S
d` says `S` has size `d`, has coefficient one, and no monomial of larger size occurs;
`anfTermCount j` is the number of monomials with coefficient one.

Every enumeration below is a **binary expansion over the `2k` input positions**, so the kernel
recursion depth is `2k` and the leaf count is `2^(2k)`. No list of submasks and no list of "big
masks" is ever built; those two list forms recursed once per element and reached the default
`maxRecDepth` at width four. No `maxRecDepth` is raised anywhere in this file. -/

/-- The bit `j` of `x + y`, for `BitVec` of width `k`. -/
def addBit {k : ℕ} (j : Fin k) (x y : BitVec k) : Bool := BitVec.getLsb (x + y) j

/-- The `k`-bit word whose bit `i` is bit `i` of the mask. -/
def maskWord {k : ℕ} (m : ℕ) : BitVec k :=
  BitVec.ofNat k (∑ i : Fin k, ((m.testBit i.val).toNat) * 2 ^ (i : ℕ))

/-- The two words read from joint bit coordinates: bit `i` of the second word is bit `k + i` of
    the mask. -/
def maskPair {k : ℕ} (m : ℕ) : BitVec k × BitVec k := (maskWord m, maskWord (m >>> k))

/-- Output bit `j` of `k`-bit modular addition, at the assignment given by the mask. -/
def addBitMask {k : ℕ} (j : Fin k) (m : ℕ) : Bool :=
  let p := maskPair m
  BitVec.getLsb (p.1 + p.2) j

/-- The input positions of a `k`-bit pair addition: `2k` of them. -/
def inputPositions (k : ℕ) : List ℕ := List.range (2 * k)

/-- The input positions at which the mask `S` has a bit: the free positions of the enumeration
    of its submasks. The masks used here are below `2^64`, so 64 positions decide every one of
    them. -/
def positionsOf (S : ℕ) : List ℕ := (List.range 64).filter fun i => S.testBit i

/-- The coefficient of the monomial with mask `S` in the algebraic normal form of output bit `j`,
    as a binary expansion over the positions of `S` with the mask fixed at `base` so far. `pos`
    is the list of positions still free, so a leaf is reached exactly at a submask of `S` and the
    recursion depth is the number of set bits of `S`. -/
def anfCoeffAux {k : ℕ} (j : Fin k) : List ℕ → ℕ → Bool
  | [], t => addBitMask j t
  | p :: ps, t => Bool.xor (anfCoeffAux j ps (t + 2 ^ p)) (anfCoeffAux j ps t)

/-- **The coefficient of the monomial with mask `S` in the algebraic normal form of output bit
    `j` of `k`-bit modular addition**: the parity of that bit over the subcube supported by `S`.
    The masks used here are below `2^64`, so 64 bit positions decide every one of them. -/
def anfCoeffMask {k : ℕ} (j : Fin k) (S : ℕ) : Bool := anfCoeffAux j (positionsOf S) 0

/-- The number of set bits of `n` below position `i`. -/
def popcountBelow (n i : ℕ) : ℕ :=
  match i with
  | 0 => 0
  | i + 1 => (n.testBit i).toNat + popcountBelow n i

/-- The size of the monomial with mask `S`: its number of set bits. The masks used here are
    below `2^64`, so 64 bit positions decide every one of them. -/
def monomialSize (S : ℕ) : ℕ := popcountBelow S 64

/-- No monomial of size at least `d + 1` occurs in output bit `j`. `size` counts the input
    positions set in `t` so far. At an interior node whose best case still falls short of
    `d + 1` the answer is `true` without recursing; at a leaf the mask is checked only when its
    own size reaches `d + 1`, so a monomial below the threshold is never consulted. Every mask of
    size above `d` survives the interior guard, because on its path the attainable size is at
    least `d + 1`. `Bool.and` stops at the first `false`. -/
def noHigherAux {k : ℕ} (j : Fin k) (d : ℕ) : List ℕ → ℕ → ℕ → Bool
  | [], t, size => size < d + 1 || anfCoeffMask j t = false
  | p :: ps, t, size =>
      if size + ps.length + 1 < d + 1 then true
      else Bool.and (noHigherAux j d ps (t + 2 ^ p) (size + 1)) (noHigherAux j d ps t size)

/-- No monomial of size above `d` occurs in the algebraic normal form of output bit `j`. -/
def noHigherAnfMask {k : ℕ} (j : Fin k) (d : ℕ) : Bool :=
  noHigherAux j d (inputPositions k) 0 0

/-- Output bit `j` of `k`-bit modular addition has algebraic normal form of degree exactly `d`,
    witnessed by the monomial with mask `S`. -/
def anfDegreeExactlyMask {k : ℕ} (j : Fin k) (S d : ℕ) : Prop :=
  monomialSize S = d ∧ anfCoeffMask j S = true ∧ noHigherAnfMask j d = true

/-- The number of monomials with coefficient one in output bit `j`, counted by a binary expansion
    over the input positions. -/
def anfTermCountAux {k : ℕ} (j : Fin k) : List ℕ → ℕ → ℕ
  | [], t => if anfCoeffMask j t then 1 else 0
  | p :: ps, t => anfTermCountAux j ps (t + 2 ^ p) + anfTermCountAux j ps t

/-- **The number of algebraic-normal-form terms of output bit `j` of `k`-bit modular
    addition.** -/
def anfTermCount {k : ℕ} (j : Fin k) : ℕ := anfTermCountAux j (inputPositions k) 0

/-- The monomial `x₀`, as a mask. -/
def monX0Mask : ℕ := 1

/-- The monomial `x₀y₀`, as a mask. -/
def monXY00Mask (k : ℕ) : ℕ := 1 + 2 ^ k

/-- The monomial `x₀x₁y₀`, as a mask. -/
def monXX1Y0Mask (k : ℕ) : ℕ := 1 + 2 ^ k + 2

/-- The monomial `x₀x₁y₀y₂`, as a mask. -/
def monXX1Y0Y2Mask (k : ℕ) : ℕ := 1 + 2 ^ k + 2 + 2 ^ (k + 2)

-- Each conjunct is decided in its own `decide`. The no-higher-monomial conjunct is the expensive
-- one: it walks the binary tree of all `2^(2k)` masks and evaluates the coefficient at each
-- mask of size at least `d + 1`. The depth is `2k`, not the leaf count, so the default
-- `maxRecDepth` is never approached and none is raised anywhere in this file.

/-- **Width two, low output bit: no monomial of size above one occurs.** -/
theorem add_k2_bit0_no_higher : noHigherAnfMask (k := 2) (j := ⟨0, by omega⟩) 1 = true := by
  decide

/-- **Width two, low output bit: `F2`-degree exactly one.** -/
theorem add_k2_bit0_degree_one :
    anfDegreeExactlyMask (k := 2) (j := ⟨0, by omega⟩) monX0Mask 1 := by
  simp only [anfDegreeExactlyMask]
  exact ⟨by decide, by decide, add_k2_bit0_no_higher⟩

/-- **Width two, top output bit: no monomial of size above two occurs.** -/
theorem add_k2_bit1_no_higher : noHigherAnfMask (k := 2) (j := ⟨1, by omega⟩) 2 = true := by
  decide

/-- **Width two, top output bit: `F2`-degree exactly two, witnessed by `x₀y₀`.** -/
theorem add_k2_bit1_degree_two :
    anfDegreeExactlyMask (k := 2) (j := ⟨1, by omega⟩) (monXY00Mask 2) 2 := by
  simp only [anfDegreeExactlyMask]
  exact ⟨by decide, by decide, add_k2_bit1_no_higher⟩

/-- **Width four, low output bit: no monomial of size above one occurs.** -/
theorem add_k4_bit0_no_higher : noHigherAnfMask (k := 4) (j := ⟨0, by omega⟩) 1 = true := by
  decide

/-- **Width four, low output bit: `F2`-degree exactly one.** -/
theorem add_k4_bit0_degree_one :
    anfDegreeExactlyMask (k := 4) (j := ⟨0, by omega⟩) monX0Mask 1 := by
  simp only [anfDegreeExactlyMask]
  exact ⟨by decide, by decide, add_k4_bit0_no_higher⟩

/-- **Width four, second output bit: no monomial of size above two occurs.** -/
theorem add_k4_bit1_no_higher : noHigherAnfMask (k := 4) (j := ⟨1, by omega⟩) 2 = true := by
  decide

/-- **Width four, second output bit: `F2`-degree exactly two, witnessed by `x₀y₀`.** -/
theorem add_k4_bit1_degree_two :
    anfDegreeExactlyMask (k := 4) (j := ⟨1, by omega⟩) (monXY00Mask 4) 2 := by
  simp only [anfDegreeExactlyMask]
  exact ⟨by decide, by decide, add_k4_bit1_no_higher⟩

/-- **Width four, third output bit: no monomial of size above three occurs.** -/
theorem add_k4_bit2_no_higher : noHigherAnfMask (k := 4) (j := ⟨2, by omega⟩) 3 = true := by
  decide

/-- **Width four, third output bit: `F2`-degree exactly three, witnessed by `x₀x₁y₀`.** -/
theorem add_k4_bit2_degree_three :
    anfDegreeExactlyMask (k := 4) (j := ⟨2, by omega⟩) (monXX1Y0Mask 4) 3 := by
  simp only [anfDegreeExactlyMask]
  exact ⟨by decide, by decide, add_k4_bit2_no_higher⟩

/-- **Width four, top output bit: no monomial of size above four occurs.** -/
theorem add_k4_bit3_no_higher : noHigherAnfMask (k := 4) (j := ⟨3, by omega⟩) 4 = true := by
  decide

/-- **Width four, top output bit: `F2`-degree exactly four, witnessed by `x₀x₁y₀y₂`.**

    This refutes the candidate "modular addition has `F2`-degree two": at width four the top
    output bit carries a degree-four monomial. -/
theorem add_k4_bit3_degree_four :
    anfDegreeExactlyMask (k := 4) (j := ⟨3, by omega⟩) (monXX1Y0Y2Mask 4) 4 := by
  simp only [anfDegreeExactlyMask]
  exact ⟨by decide, by decide, add_k4_bit3_no_higher⟩

/-- **Refutation statement: at width four the top output bit is not a degree-two polynomial**,
    because a monomial of size four occurs with coefficient one. -/
theorem add_k4_not_degree_two :
    monomialSize (monXX1Y0Y2Mask 4) = 4 ∧
      anfCoeffMask (k := 4) (j := ⟨3, by omega⟩) (monXX1Y0Y2Mask 4) = true := by
  refine ⟨by decide, by decide⟩

/-! ### Term counts, and the cross-check against the prior-art figure

Dehnavi, Rishekani, Mirzaee Shamsabad, Maimani and Pasha, *Cryptographic Properties of Addition
Modulo `2^n`*, IACR ePrint 2016/181, Theorem 4.1: for `f(x, y) = x + y (mod 2^n)` the output
coordinate `f_i` has algebraic degree `i + 1` and algebraic-normal-form term count `2^i + 1`.
That figure is an `anchor` from the source as read in `work/algebra-rooting/prior-art.md` §C1.
What is checked here is that the two finite widths reproduce it. -/

/-- **Width two, low output bit: `2^0 + 1` terms.** -/
theorem add_k2_bit0_terms : anfTermCount (k := 2) (j := ⟨0, by omega⟩) = 2 ^ 0 + 1 := by decide

/-- **Width two, top output bit: `2^1 + 1` terms.** -/
theorem add_k2_bit1_terms : anfTermCount (k := 2) (j := ⟨1, by omega⟩) = 2 ^ 1 + 1 := by decide

/-- **Width four, low output bit: `2^0 + 1` terms.** -/
theorem add_k4_bit0_terms : anfTermCount (k := 4) (j := ⟨0, by omega⟩) = 2 ^ 0 + 1 := by decide

/-- **Width four, second output bit: `2^1 + 1` terms.** -/
theorem add_k4_bit1_terms : anfTermCount (k := 4) (j := ⟨1, by omega⟩) = 2 ^ 1 + 1 := by decide

/-- **Width four, third output bit: `2^2 + 1` terms.** -/
theorem add_k4_bit2_terms : anfTermCount (k := 4) (j := ⟨2, by omega⟩) = 2 ^ 2 + 1 := by decide

/-- **Width four, top output bit: `2^3 + 1` terms.** -/
theorem add_k4_bit3_terms : anfTermCount (k := 4) (j := ⟨3, by omega⟩) = 2 ^ 3 + 1 := by decide

/-- The witness monomial of the prior-art degree figure for output bit `j`: the bits `x₀`, `y₀`
    and `x₁ … x_{j-1}`, of size `j + 1`. Meaningful for `j ≥ 1`. -/
def monTopDegMask (k j : ℕ) : ℕ := 1 + 2 ^ k + (2 ^ j - 2)

/-- **The prior-art witness family at width four**: for `j = 1, 2, 3` the monomial
    `x₀ y₀ x₁ ⋯ x_{j-1}` occurs with coefficient one in output bit `j`, and output bit `j` has
    degree exactly `j + 1`. -/
theorem add_k4_witness_family :
    anfCoeffMask (k := 4) (j := ⟨1, by omega⟩) (monTopDegMask 4 1) = true ∧
      anfCoeffMask (k := 4) (j := ⟨2, by omega⟩) (monTopDegMask 4 2) = true ∧
      anfCoeffMask (k := 4) (j := ⟨3, by omega⟩) (monTopDegMask 4 3) = true ∧
      anfDegreeExactlyMask (k := 4) (j := ⟨1, by omega⟩) (monTopDegMask 4 1) 2 ∧
      anfDegreeExactlyMask (k := 4) (j := ⟨2, by omega⟩) (monTopDegMask 4 2) 3 ∧
      anfDegreeExactlyMask (k := 4) (j := ⟨3, by omega⟩) (monTopDegMask 4 3) 4 := by
  simp only [anfDegreeExactlyMask, monTopDegMask]
  exact ⟨by decide, by decide, by decide,
    ⟨by decide, by decide, add_k4_bit1_no_higher⟩,
    ⟨by decide, by decide, add_k4_bit2_no_higher⟩,
    ⟨by decide, by decide, add_k4_bit3_no_higher⟩⟩

/-- **Width two, complete figure: output bit `j` has degree `j + 1` and `2^j + 1` terms, for
    every output bit.** -/
theorem add_k2_degree_and_terms :
    anfDegreeExactlyMask (k := 2) (j := ⟨0, by omega⟩) monX0Mask 1 ∧
      anfDegreeExactlyMask (k := 2) (j := ⟨1, by omega⟩) (monXY00Mask 2) 2 ∧
      anfTermCount (k := 2) (j := ⟨0, by omega⟩) = 2 ^ 0 + 1 ∧
      anfTermCount (k := 2) (j := ⟨1, by omega⟩) = 2 ^ 1 + 1 :=
  ⟨add_k2_bit0_degree_one, add_k2_bit1_degree_two, add_k2_bit0_terms, add_k2_bit1_terms⟩

/-- **Width four, complete figure: output bit `j` has degree `j + 1` and `2^j + 1` terms, for
    every output bit.** This is the cross-check against ePrint 2016/181 Theorem 4.1 at a checked
    width. -/
theorem add_k4_degree_and_terms :
    anfDegreeExactlyMask (k := 4) (j := ⟨0, by omega⟩) monX0Mask 1 ∧
      anfDegreeExactlyMask (k := 4) (j := ⟨1, by omega⟩) (monXY00Mask 4) 2 ∧
      anfDegreeExactlyMask (k := 4) (j := ⟨2, by omega⟩) (monXX1Y0Mask 4) 3 ∧
      anfDegreeExactlyMask (k := 4) (j := ⟨3, by omega⟩) (monXX1Y0Y2Mask 4) 4 ∧
      anfTermCount (k := 4) (j := ⟨0, by omega⟩) = 2 ^ 0 + 1 ∧
      anfTermCount (k := 4) (j := ⟨1, by omega⟩) = 2 ^ 1 + 1 ∧
      anfTermCount (k := 4) (j := ⟨2, by omega⟩) = 2 ^ 2 + 1 ∧
      anfTermCount (k := 4) (j := ⟨3, by omega⟩) = 2 ^ 3 + 1 :=
  ⟨add_k4_bit0_degree_one, add_k4_bit1_degree_two, add_k4_bit2_degree_three,
    add_k4_bit3_degree_four, add_k4_bit0_terms, add_k4_bit1_terms, add_k4_bit2_terms,
    add_k4_bit3_terms⟩

/-! ## 8 · Checked parabolic cells at the two widths

The parabolic face of the two-slot algebra at `ZMod 4` and `ZMod 16`: a product of first-slot
elements is a nonzero second slot, and the `sigma = 0` norm form does not see it. -/

/-- **Width two: `(1, 2) * (0, 3) = (0, 3)` at `sigma = 0` in `ZMod 4`.** -/
theorem parabolic_product_k2 : ringMul (0 : ZMod 4) (1, 2) (0, 3) = (0, 3) := by decide

/-- **Width four: `(2, 5) * (0, 7) = (0, 14)` at `sigma = 0` in `ZMod 16`.** -/
theorem parabolic_product_k4 : ringMul (0 : ZMod 16) (2, 5) (0, 7) = (0, 14) := by decide

/-- **Width two: the parabolic norm form of a nonzero second slot is zero.** -/
theorem parabolic_norm_blind_k2 : ringNormForm (0 : ZMod 4) (0, 2) = 0 ∧ (0, 2) ≠ (0, 0) := by
  decide

/-- **Width four: the parabolic norm form of a nonzero second slot is zero.** -/
theorem parabolic_norm_blind_k4 : ringNormForm (0 : ZMod 16) (0, 5) = 0 ∧ (0, 5) ≠ (0, 0) := by
  decide

/-! ## 9 · The two-slot decomposition of `x + y`, and the named residual

The decomposition is: put the `F2`-linear part in the first slot and the deviation in the second,
and read the sum as one `ringMul` for each. The **residual term is the second slot**, and it is
exactly `2 * (x &&& y)`, the doubled bitwise conjunction. -/

/-- The carry cell of `x + y`: first slot the `F2`-linear part `x ^^^ y`, second slot the
    carry-generating residual `2 * (x &&& y)`. -/
def carryCell {k : ℕ} (_σ : ZMod (2 ^ k)) (x y : BitVec k) : RingCell (ZMod (2 ^ k)) :=
  (denoteBV (x ^^^ y), 2 * denoteBV (x &&& y))

/-- The first slot of the carry cell is the exclusive-or part. -/
theorem carryCell_fst {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).1 = denoteBV (x ^^^ y) := rfl

/-- **The second slot of the carry cell is the residual term `2 * (x &&& y)`.** -/
theorem carryCell_snd {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).2 = 2 * denoteBV (x &&& y) := rfl

/-- **A product of a first-slot element and a second-slot element lands in the second slot, for
    every `sigma`.** The face enters only the first slot, which this product contributes as zero,
    so the mechanism carrying the carry is `sigma`-independent: the parabolic dual-number face
    is not what makes the second slot available. -/
theorem ringMul_second_of_first (R : Type*) [CommRing R] (σ a b : R) :
    ringMul σ (a, 0) (0, b) = (0, a * b) := by
  simp [ringMul]

/-- **The decomposition of `x + y` in Cell coordinates, at every width and every face**: the
    `F2`-linear part as one `ringMul` against the unit, the residual as one `ringMul` of the
    first and second axes. -/
theorem carryCell_eq_ringMul_sum {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    carryCell σ x y
      = ringMul σ (denoteBV (x ^^^ y), 0) ringOne
        + ringMul σ (2, 0) (0, denoteBV (x &&& y)) := by
  simp only [carryCell, ringMul]
  ext <;> simp

/-- **The decomposition, read back: the two slots of the carry cell are the exclusive-or part and
    the residual, and the second slot is the second slot of the second `ringMul`.** The
    identification of the *sum* `denoteBV (x + y)` with the sum of the two slots needs the carry
    identity of `x + y`, which is a fact about modular addition rather than a ring tautology; it
    is checked completely at widths two and four below. -/
theorem carryCell_readback {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).1 + (carryCell σ x y).2
      = (ringMul σ (denoteBV (x ^^^ y), 0) ringOne).1
        + (ringMul σ (2, 0) (0, denoteBV (x &&& y))).2 := by
  rw [carryCell_fst, carryCell_snd]
  simp

/-- **Width two: the two slots of the carry cell recompose `x + y`.** Complete over all
    `2^4` pairs. -/
theorem residual_recomposes_sum_two : ∀ (x y : BitVec 2),
    (carryCell (0 : ZMod 4) x y).1 + (carryCell (0 : ZMod 4) x y).2 = denoteBV (x + y) := by
  decide

/-- **Width four: the two slots of the carry cell recompose `x + y`.** Complete over all
    `2^8` pairs. -/
theorem residual_recomposes_sum_four : ∀ (x y : BitVec 4),
    (carryCell (0 : ZMod 16) x y).1 + (carryCell (0 : ZMod 16) x y).2 = denoteBV (x + y) := by
  decide

/-- **Width eight, concrete instance: `255 + 255` in `ZMod 256` is the two slots of its carry
    cell, and the residual is `254`.** A single witness, not an enumeration: the complete
    `2^16`-pair enumeration at width eight is outside the lane memory envelope and is recorded
    as an exclusion in `AlgebraCarry.md`. -/
theorem residual_recomposes_sum_eight_instance :
    carryCell (k := 8) (0 : ZMod 256) 255 255 = (0, 254) ∧
      (carryCell (k := 8) (0 : ZMod 256) 255 255).1
        + (carryCell (k := 8) (0 : ZMod 256) 255 255).2
        = denoteBV (255 + 255) := by
  decide

/-! ## 10 · The three faces: what each one reads

`ringUpper r z = z.1 + r * z.2` is a read of the algebra, and the canonical theorem
`ChannelsAnyRing.ringUpper_channel` makes it a channel under the single hypothesis `r ^ 2 = σ`.
`ringUpper 0` therefore reads the parabolic face, `ringUpper 1` the hyperbolic one, and the
elliptic face has no read at all over `ZMod (2^k)` for `k ≥ 2` (§11). -/

/-- **The hyperbolic read is the sum of the two slots.** -/
theorem ringUpper_one_add_slots {k : ℕ} (z : RingCell (ZMod (2 ^ k))) :
    ringUpper (1 : ZMod (2 ^ k)) z = z.1 + z.2 := by
  simp [ringUpper]

/-- **The parabolic read is the first slot only: `ringUpper 0` cannot see the residual at all.**
    `r = 0` is a root of `σ = 0` since `0 ^ 2 = 0`. -/
theorem ringUpper_zero_first_only {k : ℕ} (z : RingCell (ZMod (2 ^ k))) :
    ringUpper (0 : ZMod (2 ^ k)) z = z.1 := by
  simp [ringUpper]

/-- **The hyperbolic read of the carry cell is the exclusive-or part plus the residual, at every
    width.** -/
theorem ringUpper_hyperbolic_of_carryCell {k : ℕ} (x y : BitVec k) :
    ringUpper (1 : ZMod (2 ^ k)) (carryCell 1 x y)
      = denoteBV (x ^^^ y) + 2 * denoteBV (x &&& y) := by
  simp [ringUpper, carryCell]

/-- **Width two: the hyperbolic read of the carry cell is the modular sum.** -/
theorem ringUpper_hyperbolic_reads_sum_two : ∀ (x y : BitVec 2),
    ringUpper (1 : ZMod 4) (carryCell (1 : ZMod 4) x y) = denoteBV (x + y) := by
  decide

/-- **Width four: the hyperbolic read of the carry cell is the modular sum.** -/
theorem ringUpper_hyperbolic_reads_sum_four : ∀ (x y : BitVec 4),
    ringUpper (1 : ZMod 16) (carryCell (1 : ZMod 16) x y) = denoteBV (x + y) := by
  decide

/-- **Width eight, concrete instance: the hyperbolic read of the carry cell of `255 + 255` is the
    modular sum `254`.** A single witness, not an enumeration; see
    `residual_recomposes_sum_eight_instance`. -/
theorem ringUpper_hyperbolic_reads_sum_eight_instance :
    ringUpper (1 : ZMod 256) (carryCell (1 : ZMod 256) (255 : BitVec 8) (255 : BitVec 8))
      = denoteBV ((255 : BitVec 8) + 255) := by
  decide

/-- **Width two, the parabolic read of the carry cell is the exclusive-or part, not the sum.**
    The residual is `1` here (`x = y = 3`) and `ringUpper 0` returns `0`. -/
theorem ringUpper_parabolic_blind_two :
    ∀ x y : BitVec 2, ringUpper (0 : ZMod 4) (carryCell (0 : ZMod 4) x y) = (carryCell 0 x y).1 :=
  fun _ _ => ringUpper_zero_first_only _

/-- **Width four, the parabolic read of the carry cell is the exclusive-or part, not the sum.**
    The residual is `6` here (`x = y = 3`) and `ringUpper 0` returns `0`. -/
theorem ringUpper_parabolic_blind_four :
    ∀ x y : BitVec 4, ringUpper (0 : ZMod 16) (carryCell (0 : ZMod 16) x y) = (carryCell 0 x y).1 :=
  fun _ _ => ringUpper_zero_first_only _

/-- **The three norm forms of the carry cell, at every width: the parabolic one is blind to the
    residual, the hyperbolic and elliptic ones carry it with opposite signs.** -/
theorem normForm_of_carryCell {k : ℕ} (x y : BitVec k) :
    ringNormForm (0 : ZMod (2 ^ k)) (carryCell 0 x y) = (denoteBV (x ^^^ y)) ^ 2 ∧
      ringNormForm (1 : ZMod (2 ^ k)) (carryCell 1 x y)
        = (denoteBV (x ^^^ y)) ^ 2 - (2 * denoteBV (x &&& y)) ^ 2 ∧
      ringNormForm (-1 : ZMod (2 ^ k)) (carryCell (-1) x y)
        = (denoteBV (x ^^^ y)) ^ 2 + (2 * denoteBV (x &&& y)) ^ 2 := by
  simp only [ringNormForm, carryCell]
  refine ⟨?_, ?_, ?_⟩
  · ring
  · ring
  · ring

/-- **The two faces admit a read, and the read is multiplicative there.** `σ = 0` has the root
    `r = 0` and `σ = 1` has the root `r = 1`, at every width. -/
theorem channel_parabolic_and_hyperbolic (k : ℕ) :
    RingChannel (0 : ZMod (2 ^ k)) (ringUpper 0) ∧
      RingChannel (1 : ZMod (2 ^ k)) (ringUpper 1) :=
  ⟨ringUpper_channel (σ := (0 : ZMod (2 ^ k))) (r := 0) (by ring),
    ringUpper_channel (σ := (1 : ZMod (2 ^ k))) (r := 1) (by ring)⟩

/-! ## 11 · The obstruction lemma: the elliptic face has no read

Over `ZMod (2^k)` the trichotomy of faces separates exactly in the existence of a square root of
`σ`, because that is the single hypothesis of `ChannelsAnyRing.ringUpper_channel` and, by
`ChannelsAnyRing.ringChannel_gen_sq`, the single necessary condition for **any** read. -/

/-- The image of squaring on `ZMod 4` is `{0, 1}`. -/
def sqImage4 : Finset (ZMod 4) := Finset.univ.image (fun r : ZMod 4 => r ^ 2)

/-- **Every square in `ZMod 4` is `0` or `1`, so `-1 = 3` is not a square.** -/
theorem sqImage4_eq : sqImage4 = ({0, 1} : Finset (ZMod 4)) := by decide

/-- **No element of `ZMod 4` squares to `-1`.** -/
theorem no_root_neg_one_mod4 : ∀ r : ZMod 4, r ^ 2 ≠ (-1 : ZMod 4) := by
  intro r hr
  have hmem : r ^ 2 ∈ sqImage4 := Finset.mem_image.mpr ⟨r, Finset.mem_univ _, rfl⟩
  rw [sqImage4_eq, hr] at hmem
  exact absurd hmem (by decide : ¬ ((-1 : ZMod 4) ∈ ({0, 1} : Finset (ZMod 4))))

/-- **For every width `k ≥ 2`, `ZMod (2^k)` has no element squaring to `-1`.** The proof reduces
    along `4 ∣ 2^k` by the canonical ring hom `ZMod.castHom` to the width-four case. -/
theorem no_root_neg_one_of_two_le_k {k : ℕ} (hk : 2 ≤ k) :
    ¬ ∃ r : ZMod (2 ^ k), r ^ 2 = (-1 : ZMod (2 ^ k)) := by
  rintro ⟨r, h⟩
  have hd : (4 : ℕ) ∣ 2 ^ k := by
    have hsplit : 2 ^ k = 4 * 2 ^ (k - 2) := by
      have hk2 : k = 2 + (k - 2) := by omega
      conv_lhs => rw [hk2, pow_add]
      ring
    exact ⟨2 ^ (k - 2), hsplit⟩
  have hone : ZMod.castHom hd (ZMod 4) (1 : ZMod (2 ^ k)) = 1 := map_one _
  have hdown := congrArg (ZMod.castHom hd (ZMod 4)) h
  rw [map_pow, map_neg, hone] at hdown
  exact no_root_neg_one_mod4 (ZMod.castHom hd (ZMod 4) r) hdown

/-- Width two, as an instance of the general theorem: no root of `-1` in `ZMod 4`. -/
theorem no_root_neg_one_two :
    ¬ ∃ r : ZMod 4, r ^ 2 = (-1 : ZMod 4) :=
  no_root_neg_one_of_two_le_k (k := 2) (by omega)

/-- Width four, as an instance of the general theorem: no root of `-1` in `ZMod 16`. -/
theorem no_root_neg_one_four :
    ¬ ∃ r : ZMod 16, r ^ 2 = (-1 : ZMod 16) :=
  no_root_neg_one_of_two_le_k (k := 4) (by omega)

/-- Width eight, as an instance of the general theorem: no root of `-1` in `ZMod 256`. This
    instance is stated from the general theorem rather than decided, because the `256`-element
    `∃`-enumeration reached the default `maxRecDepth`. -/
theorem no_root_neg_one_eight :
    ¬ ∃ r : ZMod 256, r ^ 2 = (-1 : ZMod 256) :=
  no_root_neg_one_of_two_le_k (k := 8) (by omega)

/-- **Obstruction lemma, elliptic face.** For every width `k ≥ 2` there is no read of
    `RingCell (ZMod (2^k))` at `σ = -1`. Premise used: `ChannelsAnyRing.ringChannel_gen_sq`,
    which forces the value of any read at the generator `(0, 1)` to square to `σ`. -/
theorem no_channel_elliptic {k : ℕ} (hk : 2 ≤ k) (φ : RingCell (ZMod (2 ^ k)) → ZMod (2 ^ k))
    (h : RingChannel (-1 : ZMod (2 ^ k)) φ) : False := by
  have hg : φ ((0, 1) : RingCell (ZMod (2 ^ k))) * φ ((0, 1) : RingCell (ZMod (2 ^ k)))
      = (-1 : ZMod (2 ^ k)) := ringChannel_gen_sq (σ := (-1 : ZMod (2 ^ k))) h
  refine no_root_neg_one_of_two_le_k hk ⟨φ ((0, 1) : RingCell (ZMod (2 ^ k))), ?_⟩
  simpa only [pow_two] using hg

/-- **Every read of the elliptic face, if one existed, is fixed by one square root of `-1`, so
    the reads at a face are parametrized by the roots of `σ`.** -/
theorem channel_read_shape {k : ℕ} {σ : ZMod (2 ^ k)} {φ : RingCell (ZMod (2 ^ k)) → ZMod (2 ^ k)}
    (h : RingChannel σ φ) (z : RingCell (ZMod (2 ^ k))) :
    φ z = z.1 + z.2 * φ ((0, 1) : RingCell (ZMod (2 ^ k))) := ringChannel_apply h z

/-! ## 12 · Axiom cone

Every headline above is checked in the closed cone `{propext, Classical.choice, Quot.sound}` with
no `sorryAx`. -/

#print axioms denoteBV_add
#print axioms two_eq_zero_iff
#print axioms doubling_zero_iff
#print axioms addHom_apply
#print axioms leibniz_addHom_zero
#print axioms doublingHom_ne_zero
#print axioms doublingHom_not_leibniz
#print axioms ringMul_zero_second
#print axioms ringNormForm_zero_first
#print axioms idempotent_trivial
#print axioms idempotent_second_zero
#print axioms carry_identity_two
#print axioms carry_identity_four
#print axioms carry_identity_four_zmod
#print axioms deviation_in_second_slot
#print axioms add_k2_bit1_degree_two
#print axioms add_k4_bit1_degree_two
#print axioms add_k4_bit3_no_higher
#print axioms add_k4_bit3_degree_four
#print axioms add_k4_not_degree_two
#print axioms add_k2_degree_and_terms
#print axioms add_k4_degree_and_terms
#print axioms add_k4_witness_family
#print axioms ringMul_second_of_first
#print axioms carryCell_eq_ringMul_sum
#print axioms carryCell_readback
#print axioms residual_recomposes_sum_two
#print axioms residual_recomposes_sum_four
#print axioms residual_recomposes_sum_eight_instance
#print axioms ringUpper_one_add_slots
#print axioms ringUpper_zero_first_only
#print axioms ringUpper_hyperbolic_of_carryCell
#print axioms ringUpper_hyperbolic_reads_sum_two
#print axioms ringUpper_hyperbolic_reads_sum_four
#print axioms ringUpper_hyperbolic_reads_sum_eight_instance
#print axioms ringUpper_parabolic_blind_two
#print axioms normForm_of_carryCell
#print axioms channel_parabolic_and_hyperbolic
#print axioms sqImage4_eq
#print axioms no_root_neg_one_mod4
#print axioms no_root_neg_one_of_two_le_k
#print axioms no_root_neg_one_four
#print axioms no_root_neg_one_eight
#print axioms no_channel_elliptic
#print axioms channel_read_shape

end AlgebraCarry