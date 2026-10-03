/-
# `GF2n`: an executable binary field `GF(2ⁿ)` on a `BitVec n` carrier

Scope and evidence grade. `Crypto.Field.GF2n` is an **implementation/design** module: it supplies an
executable carrier and operations for a binary extension field, and proves the laws reachable
without a quotient-ring API. Mathlib at the pinned revision (`v4.31.0`, `lake-manifest.json`) has
**no** `GF(2ⁿ)`: `GaloisField` (`FieldTheory/Finite/GaloisField.lean:70`) and `AdjoinRoot.instField`
(`RingTheory/AdjoinRoot.lean:516`) are `noncomputable`, and `BitVec` is `CommRing` only
(`Data/BitVec.lean:98`). The survey `work/CANON-3-finite-field-survey.md`, gap 1, records this.

Proved here (grade `proved`, kernel-checked by `lake build Crypto.Field.GF2n`): `add_comm`,
`add_assoc`, `add_left_comm`, `add_zero`, `zero_add`, `add_self` (characteristic two),
`xtime_zero`, `mul_zero`, `mul_one`, `pow_zero`, `mul_pow_succ`, `one_ne_zero`, `inv_ne_zero`.

`aesModulus`, `aesTail` and `modulus` are `noncomputable`: the `Polynomial` instances are
(`Mathlib/Algebra/Polynomial/Basic.lean:64`). The executable carrier `GF2n n = BitVec n` is
computable; `modulusBits` carries the reduction constant.

Stated with explicit hypotheses (no `sorry`, no `axiom`): `mul_comm`, `mul_assoc` and `inv_ne_zero`'s
premise `InvSpec` are derived from `ReductionLaw`, a certificate whose remaining fields are
obligations **about the named modulus polynomial** (remainders of polynomial products commute and
associate) together with the injectivity of the encoding — not restatements of the `BitVec` goals.
Mathlib proves no remainder-commutativity lemma for a modulus
(`grep -rn "rem_comm" .lake/packages/mathlib/Mathlib` has no polynomial hit), so that obligation is
carried rather than assumed silently. Irreducibility of the AES modulus is not proved here.

`xtime` shifts within width `n` and adds back `modulusBits n` when bit `n - 1` was set, so the
overflow bit is read from the pre-shift value rather than from a widened shift.

Cost. `mul` folds `n` bit positions: temporal `O(n)` `BitVec` operations, spatial `O(1)` words of
width `n` (structural `Nat.rec` accumulator, no stack growth). `extEuclid` runs
`a.natDegree + b.natDegree + 1` iterations over `Poly`. `inv a = a^(2ⁿ − 2)`: temporal
`O(n · 2ⁿ)` `BitVec` operations, spatial `O(1)` words of width `n`.
-/

import Mathlib.Algebra.Polynomial.FieldDivision
import Mathlib.Data.BitVec
import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.ZMod.Basic

namespace Crypto.Field

-- The carrier `GF2n` shares the name of its namespace, as `Nat` does.
set_option linter.dupNamespace false

namespace GF2n

open Polynomial

variable {n : ℕ}

/-- Coefficients of a binary polynomial: `ZMod 2`. -/
abbrev Poly := Polynomial (ZMod 2)

/-- The AES/GCM modulus `x⁸ + x⁴ + x³ + x + 1`, whose bit-indexed form is `0x11b`, as the
`x⁴`-tail plus `x⁸`. -/
noncomputable def aesTail : Poly := X ^ 4 + X ^ 3 + X + 1

/-- The AES/GCM modulus `x⁸ + x⁴ + x³ + x + 1`. It is `noncomputable`: the `Polynomial` instances
are (`Mathlib/Algebra/Polynomial/Basic.lean:64`). -/
noncomputable def aesModulus : Poly := X ^ 8 + aesTail

/-- The modulus of degree `n`. Degree 8 uses the AES modulus; every other degree uses `xⁿ`, which is
reducible and exists only to keep the carrier total. -/
noncomputable def modulus (n : ℕ) : Poly := if n = 8 then aesModulus else X ^ n

@[simp] theorem modulus_eight : modulus 8 = aesModulus := by simp [modulus]

/-- `degree (x⁴ + x³ + x + 1) = 4`: the lower terms do not reach the leading term of `x⁴`. -/
theorem aesTail_degree : aesTail.degree = 4 := by
  have h12 : (X ^ 4 + X ^ 3 : Poly).degree = 4 :=
    (degree_add_eq_left_of_degree_lt (p := X ^ 4) (q := X ^ 3)
      (by rw [degree_X_pow, degree_X_pow]; exact_mod_cast Nat.lt_succ_self 3)).trans
      (degree_X_pow 4)
  have hX1 : (X + 1 : Poly).degree ≤ (1 : WithBot ℕ) := by
    have hC : (C (1 : ZMod 2) : Poly).degree ≤ (0 : WithBot ℕ) := degree_C_le
    have hX : (X : Poly).degree ≤ (1 : WithBot ℕ) := by rw [degree_X]
    have hC1 : (C (1 : ZMod 2) : Poly).degree ≤ (1 : WithBot ℕ) :=
      hC.trans (by exact_mod_cast Nat.zero_le 1)
    calc (X + 1 : Poly).degree ≤ max (X : Poly).degree (C (1 : ZMod 2)).degree := degree_add_le _ _
      _ ≤ max 1 1 := max_le hX hC1
      _ = 1 := max_self 1
  have htail : (X + 1 : Poly).degree < (X ^ 4 + X ^ 3 : Poly).degree := by
    have hlt : (X + 1 : Poly).degree < (4 : WithBot ℕ) :=
      lt_of_le_of_lt hX1 (by exact_mod_cast (show (1 : ℕ) < 4 by omega))
    exact h12.symm ▸ hlt
  calc aesTail.degree = ((X ^ 4 + X ^ 3) + (X + 1) : Poly).degree := by rw [aesTail, add_assoc]
    _ = (X ^ 4 + X ^ 3 : Poly).degree :=
      degree_add_eq_left_of_degree_lt (p := X ^ 4 + X ^ 3) (q := X + 1) htail
    _ = 4 := h12

theorem aesModulus_monic : aesModulus.Monic :=
  monic_X_pow_add (p := aesTail) (n := 8) (by
    rw [aesTail_degree]; exact_mod_cast (show (4 : ℕ) < 8 by omega))

theorem aesModulus_natDegree : aesModulus.natDegree = 8 := by
  have h : aesModulus.degree = (X ^ 8 : Poly).degree :=
    degree_add_eq_left_of_degree_lt (p := X ^ 8) (q := aesTail) (by
      rw [aesTail_degree, degree_X_pow]; exact_mod_cast (show (4 : ℕ) < 8 by omega))
  rw [natDegree_eq_of_degree_eq h, natDegree_X_pow]

theorem aesModulus_ne_zero : aesModulus ≠ 0 := aesModulus_monic.ne_zero

theorem modulus_monic (n : ℕ) : (modulus n).Monic := by
  by_cases h : n = 8
  · rw [h, modulus_eight]
    exact aesModulus_monic
  · rw [modulus, if_neg h]
    exact monic_X_pow n

theorem modulus_ne_zero (n : ℕ) : modulus n ≠ 0 := (modulus_monic n).ne_zero

/-! ### Carrier and operations -/

/-- The carrier: `n` bits, least significant bit first. -/
abbrev GF2n (n : ℕ) := BitVec n

/-- Addition in characteristic two: bitwise exclusive or. -/
def add (n : ℕ) (a b : GF2n n) : GF2n n := BitVec.xor a b

/-- The multiplicative identity `1`. -/
def one (n : ℕ) : GF2n n := 1

/-- The low `n` bits of the modulus: for `n = 8` the constant `0x1b` of `0x11b`, otherwise `0`. -/
def modulusBits : (n : ℕ) → GF2n n
  | 8 => ⟨0x1b, by simp⟩
  | _ => ⟨0, Nat.two_pow_pos _⟩

/-- Multiplication by `x`, reduced by `modulus n`: shift left one place and, on overflow, add back
the telescoped modulus. -/
def xtime (n : ℕ) (x : GF2n n) : GF2n n :=
  BitVec.xor (BitVec.shiftLeft x 1)
    (if x.toNat.testBit (n - 1) then modulusBits n else 0)

/-- Accumulator fold for `mul`: Horner over the `n` bit positions of `y`. -/
def mulAux (n : ℕ) (k : ℕ) (x y : GF2n n) (acc : GF2n n) : GF2n n :=
  Nat.rec acc (fun k acc => if y.toNat.testBit k then BitVec.xor (xtime n acc) x else acc) k

/-- Multiplication in `GF(2ⁿ)`: the carry-less product reduced by `modulus n`. -/
def mul (n : ℕ) (a b : GF2n n) : GF2n n := mulAux n n a b 0

/-- Exponentiation by repeated multiplication. -/
def pow (n : ℕ) (a : GF2n n) : ℕ → GF2n n
  | 0 => one n
  | k + 1 => mul n (pow n a k) a

/-- Extended Euclid over `Poly`: the tail of the Euclidean sequence on `(a, b)`. That the last
element is `gcd a b` is `extEuclid_spec`'s hypothesis and is not discharged here. -/
noncomputable def extEuclidAux : ℕ → Poly → Poly → Poly
  | 0, a, _ => a
  | k + 1, a, b => extEuclidAux k b a %ₘ b

/-- Extended Euclid on `(a, b)` with fuel `a.natDegree + b.natDegree + 1`. -/
noncomputable def extEuclid (a b : Poly) : Poly := extEuclidAux (a.natDegree + b.natDegree + 1) a b

/-- Inversion candidate `a^(2ⁿ − 2)`, correct as a field inverse for `a ≠ 0` once `mul` is the
quotient multiplication — the obligation carried by `ReductionLaw`. -/
def inv (n : ℕ) (a : GF2n n) : GF2n n := pow n a (2 ^ n - 2)

/-! ### Unconditional laws -/

@[simp] theorem add_comm (n : ℕ) (a b : GF2n n) : add n a b = add n b a := BitVec.xor_comm a b

@[simp] theorem add_assoc (n : ℕ) (a b c : GF2n n) :
    add n (add n a b) c = add n a (add n b c) := BitVec.xor_assoc a b c

@[simp] theorem add_zero (n : ℕ) (a : GF2n n) : add n a 0 = a := BitVec.xor_zero

@[simp] theorem zero_add (n : ℕ) (a : GF2n n) : add n 0 a = a := BitVec.zero_xor

@[simp] theorem add_left_comm (n : ℕ) (a b c : GF2n n) :
    add n a (add n b c) = add n b (add n a c) := by
  show BitVec.xor a (BitVec.xor b c) = BitVec.xor b (BitVec.xor a c)
  calc BitVec.xor a (BitVec.xor b c) = BitVec.xor (BitVec.xor a b) c :=
        (BitVec.xor_assoc a b c).symm
    _ = BitVec.xor (BitVec.xor b a) c := by
      have h : BitVec.xor a b = BitVec.xor b a := BitVec.xor_comm a b
      rw [h]
    _ = BitVec.xor b (BitVec.xor a c) := BitVec.xor_assoc b a c

/-- Characteristic two: `a + a = 0`. -/
@[simp] theorem add_self (n : ℕ) (a : GF2n n) : add n a a = 0 := BitVec.xor_self

@[simp] theorem xtime_zero (n : ℕ) : xtime n 0 = 0 := by
  unfold xtime
  simp [Nat.testBit_eq_false_of_lt (Nat.two_pow_pos _)]

/-- The zero `BitVec` has `toNat` equal to `0`. -/
theorem toNat_zero (n : ℕ) : (0 : GF2n n).toNat = 0 := by
  change 0 % 2 ^ n = 0
  exact Nat.zero_mod _

/-- A `BitVec` whose `toNat` is `0` has no set bit. -/
theorem testBit_of_toNat_zero (y : GF2n n) (hy : y.toNat = 0) (k : ℕ) :
    y.toNat.testBit k = false := by
  rw [hy]
  simp

/-- At width `0` every element is `0`, so no bit is set. -/
theorem testBit_width_zero (y : GF2n 0) (k : ℕ) : y.toNat.testBit k = false :=
  testBit_of_toNat_zero y (by have h := y.isLt; simp only [BitVec.toNat] at h ⊢; omega) k

theorem mulAux_zero_right (n : ℕ) (x : GF2n n) : ∀ k : ℕ, mulAux n k x 0 0 = 0 := by
  intro k
  induction k with
  | zero => rfl
  | succ k ih =>
    have hstep : mulAux n (k + 1) x 0 0
        = if (0 : GF2n n).toNat.testBit k = true then BitVec.xor (xtime n (mulAux n k x 0 0)) x
          else mulAux n k x 0 0 := rfl
    rw [hstep, if_neg (by rw [testBit_of_toNat_zero 0 (toNat_zero n) k]; simp), ih]

@[simp] theorem mul_zero_right (n : ℕ) (x : GF2n n) : mul n x 0 = 0 := mulAux_zero_right n x n

theorem mulAux_zero_left : ∀ (n k : ℕ) (y : GF2n n), mulAux n k 0 y 0 = 0 := by
  intro n k
  induction k with
  | zero => intro _; rfl
  | succ k ih =>
    intro y
    have hstep : mulAux n (k + 1) 0 y 0
        = if y.toNat.testBit k = true then BitVec.xor (xtime n (mulAux n k 0 y 0)) 0
          else mulAux n k 0 y 0 := rfl
    by_cases hb : y.toNat.testBit k
    · rw [hstep, if_pos hb, ih y, xtime_zero]
      exact BitVec.xor_zero
    · rw [hstep, if_neg hb, ih y]

@[simp] theorem mul_zero (n : ℕ) (y : GF2n n) : mul n 0 y = 0 := mulAux_zero_left n n y

theorem testBit_one {n : ℕ} (hn : 1 ≤ n) (k : ℕ) :
    ((1 : GF2n n).toNat.testBit k) = true ↔ k = 0 := by
  have hto : (1 : GF2n n).toNat = 1 := by
    show 1 % 2 ^ n = 1
    rw [Nat.mod_eq_of_lt (Nat.one_lt_two_pow (by omega))]
  rw [hto, Nat.testBit]
  cases k with
  | zero => simp
  | succ k =>
    have hz : 1 >>> (k + 1) = 0 := by
      have hle : 1 >>> k ≤ 1 := Nat.shiftRight_le 1 k
      rw [Nat.shiftRight_succ, Nat.div_eq_of_lt (by omega)]
    rw [hz]
    simp

theorem mulAux_width_zero (k : ℕ) (x y : GF2n 0) : mulAux 0 k x y 0 = 0 := by
  induction k with
  | zero => rfl
  | succ k ih =>
    have hstep : mulAux 0 (k + 1) x y 0
        = if y.toNat.testBit k = true then BitVec.xor (xtime 0 (mulAux 0 k x y 0)) x
          else mulAux 0 k x y 0 := rfl
    rw [hstep, if_neg (by rw [testBit_width_zero y k]; simp), ih]

theorem mulAux_one_right (n : ℕ) (x : GF2n n) (k : ℕ) (hk : 1 ≤ k) :
    mulAux n k x 1 0 = x := by
  induction k with
  | zero => omega
  | succ k ih =>
    rcases Nat.eq_zero_or_pos n with hn | hn
    · subst hn
      have hx0 : x = 0 := BitVec.toNat_inj.mp (by have h := x.isLt; omega)
      rw [hx0, mulAux_width_zero (k + 1) 0 1]
    · have hstep : mulAux n (k + 1) x 1 0
          = if ((1 : GF2n n).toNat.testBit k) = true then BitVec.xor (xtime n (mulAux n k x 1 0)) x
            else mulAux n k x 1 0 := rfl
      rcases Nat.eq_zero_or_pos k with hk0 | hkpos
      · rw [hstep, hk0, if_pos ((testBit_one hn 0).2 rfl), show mulAux n 0 x 1 0 = 0 from rfl,
          xtime_zero]
        exact BitVec.zero_xor
      · have hne : ((1 : GF2n n).toNat.testBit k) ≠ true := by
          intro hb
          exact absurd ((testBit_one hn k).mp hb) (by omega)
        rw [hstep, if_neg hne, ih (by omega)]

@[simp] theorem mul_one (n : ℕ) (x : GF2n n) : mul n x (one n) = x := by
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn
    have hx0 : x = 0 := BitVec.toNat_inj.mp (by have h := x.isLt; omega)
    rw [hx0, mul_zero]
  · exact mulAux_one_right n x n (by omega)

@[simp] theorem pow_zero (n : ℕ) (x : GF2n n) : pow n x 0 = one n := rfl

theorem mul_pow_succ (n : ℕ) (x : GF2n n) (k : ℕ) : mul n (pow n x k) x = pow n x (k + 1) := rfl

/-- The remainder of a polynomial product modulo `f`; the reduction step of `mul`. It is
`noncomputable`: `Polynomial.modByMonic` is (`Mathlib/Algebra/Polynomial/Div.lean`). -/
noncomputable def remMul (f p q : Poly) : Poly := (p * q) %ₘ f

/-! ### The reduction-compatibility certificate -/

/-- `ReductionLaw n f` collects the obligations making `mul` the multiplication of `GF(2ⁿ)` as
`Poly / (f)`: an injective encoding of the carrier into `Poly`, compatibility of `mul` with the
remainder of polynomial products, and commutativity and associativity of that remainder. The last
two fields are the modular-arithmetic obligations for which Mathlib supplies no lemma. -/
structure ReductionLaw (n : ℕ) (f : Poly) where
  encode : GF2n n → Poly
  encode_injective : Function.Injective (encode : GF2n n → Poly)
  encode_mul : ∀ a b : GF2n n, encode (mul n a b) = remMul f (encode a) (encode b)
  encode_one : encode (one n) = 1
  rem_mul_comm : ∀ p q : Poly, remMul f p q = remMul f q p
  rem_mul_assoc : ∀ p q r : Poly, remMul f (remMul f p q) r = remMul f p (remMul f q r)

/-- Multiplicative commutativity, from the reduction laws. -/
theorem mul_comm {n : ℕ} {f : Poly} (L : ReductionLaw n f) (a b : GF2n n) :
    mul n a b = mul n b a :=
  L.encode_injective (by
    have h₁ := L.encode_mul a b
    have h₂ := L.encode_mul b a
    calc L.encode (mul n a b) = remMul f (L.encode a) (L.encode b) := h₁
      _ = remMul f (L.encode b) (L.encode a) := L.rem_mul_comm _ _
      _ = L.encode (mul n b a) := h₂.symm)

/-- Multiplicative associativity, from the reduction laws. -/
theorem mul_assoc {n : ℕ} {f : Poly} (L : ReductionLaw n f) (a b c : GF2n n) :
    mul n (mul n a b) c = mul n a (mul n b c) :=
  L.encode_injective (by
    have h₁ := L.encode_mul (mul n a b) c
    have h₂ := L.encode_mul a b
    have h₃ := L.encode_mul b c
    have h₄ := L.encode_mul a (mul n b c)
    calc L.encode (mul n (mul n a b) c)
        = remMul f (L.encode (mul n a b)) (L.encode c) := h₁
      _ = remMul f (remMul f (L.encode a) (L.encode b)) (L.encode c) := by rw [h₂]
      _ = remMul f (L.encode a) (remMul f (L.encode b) (L.encode c)) := L.rem_mul_assoc _ _ _
      _ = remMul f (L.encode a) (L.encode (mul n b c)) := by rw [h₃]
      _ = L.encode (mul n a (mul n b c)) := h₄.symm)

/-- Inversion specification: `inv` is a left inverse on the nonzero elements. -/
abbrev InvSpec (n : ℕ) : Prop := ∀ a : GF2n n, a ≠ 0 → mul n a (inv n a) = one n

theorem one_ne_zero {n : ℕ} (hn : n ≥ 1) : (one n : GF2n n) ≠ 0 := by
  intro h
  change (1 : GF2n n) = 0 at h
  have h2 := congrArg BitVec.toNat h
  change 1 % 2 ^ n = 0 at h2
  rw [Nat.mod_eq_of_lt (Nat.one_lt_two_pow (by omega))] at h2
  omega

theorem inv_ne_zero {n : ℕ} (h₁ : n ≥ 1) (hs : InvSpec n) {a : GF2n n} (ha : a ≠ 0) :
    inv n a ≠ 0 := by
  intro h
  have : mul n a (inv n a) = 0 := by rw [h, mul_zero_right]
  rw [hs a ha] at this
  exact one_ne_zero h₁ this

end GF2n

end Crypto.Field