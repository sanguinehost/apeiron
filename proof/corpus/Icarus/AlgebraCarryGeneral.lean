/-
# Width-parametric carry identity and degree of modular addition

Re-derivation of the two central results of the carry analysis at **every** width, replacing the
per-width enumerations of `work/proofs/AlgebraCarry.lean`:

| claim | statement | form |
|---|---|---|
| the carry cell | `carryCell σ x y = (denoteBV (x ^^^ y), 2 * denoteBV (x &&& y))` | `rfl` |
| the carry cell recomposes the sum | `(carryCell σ x y).1 + (carryCell σ x y).2 = denoteBV (x + y)` | induction on the bits |
| the Cell-coordinate decomposition | `carryCell σ x y = ringMul σ (x ^^^ y, 0) ringOne + ringMul σ (2, 0) (0, x &&& y)` | algebra |
| degree of bit `j` | `totalDegree (bitPoly j) = j + 1` | induction on the bit index |

`k`, `σ`, `x`, `y` and `j` are universally quantified in every headline; no width and no face is
instantiated in a proof, and no enumeration over pairs of words appears anywhere in this file.
Nothing here is connected to SHA-2, SHA-3, ARX, a hash round or attack cost.

Grades: every declaration below is `proved` by the Lean kernel under the closed cone
`{propext, Classical.choice, Quot.sound}`; the `#print axioms` rows at the end of the file are the
record. No `sorry`, no `native_decide`, no raised `maxRecDepth`, no `maxHeartbeats 0`.
-/
import Apeiron.Trichotomy.AnyRing
import Mathlib.Algebra.MvPolynomial.Degrees
import Mathlib.Algebra.MvPolynomial.NoZeroDivisors

set_option linter.apeiron.notation false

namespace AlgebraCarryGeneral

open Apeiron.Trichotomy.AnyRing

/-! ## 1 · The notation: the exact ring and face parameters

A `k`-bit word read as an element of the modular ring `ZMod (2^k)`. -/

/-- A `k`-bit word read as an element of the modular ring `ZMod (2^k)`. -/
def denoteBV {k : ℕ} (w : BitVec k) : ZMod (2 ^ k) := (w.toNat : ZMod (2 ^ k))

/-- **Wrapping addition of `k`-bit words is addition in `ZMod (2^k)`, at every width.** -/
theorem denoteBV_add (k : ℕ) (x y : BitVec k) :
    denoteBV (x + y) = denoteBV x + denoteBV y := by
  rw [denoteBV, BitVec.toNat_add, ZMod.natCast_mod]
  exact Nat.cast_add _ _

/-! ## 2 · The natural-number carry identity, by induction on the bits

`a + b = (a ^^^ b) + 2 * (a &&& b)` is proved here from the `testBit` definition by induction on
`a`, carrying `b` along. No enumeration and no `decide` over a wide type. -/

/-- `(2 * n + b) % 2 = b % 2`. -/
theorem mod_two_two_mul_add (n b : ℕ) : (2 * n + b) % 2 = b % 2 := by omega

/-- The low bit of `2n + b` reads `b`, for `b ≤ 1`. -/
theorem testBit_two_mul_add_zero (n b : ℕ) (hb : b ≤ 1) :
    (2 * n + b).testBit 0 = decide (b = 1) := by
  rw [Nat.testBit_zero, mod_two_two_mul_add]
  obtain ⟨rfl | rfl⟩ : b = 0 ∨ b = 1 := by omega
  · rfl
  · rw [Nat.mod_eq_of_lt (by omega)]

/-- `(2 * n + b) / 2 = n` for `b ≤ 1`. -/
theorem div_two_mul_add (n b : ℕ) (hb : b ≤ 1) : (2 * n + b) / 2 = n := by omega

/-- Bit `i + 1` of `2n + b` is bit `i` of `n`, for `b ≤ 1`. -/
theorem testBit_two_mul_add_succ (n b i : ℕ) (hb : b ≤ 1) :
    (2 * n + b).testBit (i + 1) = n.testBit i := by
  rw [Nat.testBit_add_one, div_two_mul_add _ _ hb]

/-- **Bitwise exclusive-or splits over the low bit**: for `a₀ b₀ ≤ 1`,
`(2A + a₀) ^^^ (2B + b₀) = 2 * (A ^^^ B) + (a₀ ^^^ b₀)`. -/
theorem xor_two_mul_add (A B a₀ b₀ : ℕ) (ha : a₀ ≤ 1) (hb : b₀ ≤ 1) :
    (2 * A + a₀) ^^^ (2 * B + b₀) = 2 * (A ^^^ B) + (a₀ ^^^ b₀) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rcases i with _ | i
  · rw [Nat.testBit_xor, testBit_two_mul_add_zero _ _ ha, testBit_two_mul_add_zero _ _ hb,
      Nat.testBit_zero, mod_two_two_mul_add]
    have ha01 : a₀ = 0 ∨ a₀ = 1 := by omega
    have hb01 : b₀ = 0 ∨ b₀ = 1 := by omega
    rcases ha01 with ha01 | ha01 <;> rcases hb01 with hb01 | hb01 <;>
      rw [ha01, hb01] <;> rfl
  · have hsub : a₀ ^^^ b₀ ≤ 1 := by
      have hlt := Nat.xor_lt_two_pow (x := a₀) (y := b₀) (n := 1)
        (by rw [pow_one]; omega) (by rw [pow_one]; omega)
      omega
    rw [Nat.testBit_xor, testBit_two_mul_add_succ _ _ _ ha, testBit_two_mul_add_succ _ _ _ hb,
      testBit_two_mul_add_succ _ _ _ hsub, Nat.testBit_xor]

/-- **Bitwise conjunction splits over the low bit**: for `a₀ b₀ ≤ 1`,
`(2A + a₀) &&& (2B + b₀) = 2 * (A &&& B) + (a₀ &&& b₀)`. -/
theorem and_two_mul_add (A B a₀ b₀ : ℕ) (ha : a₀ ≤ 1) (hb : b₀ ≤ 1) :
    (2 * A + a₀) &&& (2 * B + b₀) = 2 * (A &&& B) + (a₀ &&& b₀) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rcases i with _ | i
  · rw [Nat.testBit_and, testBit_two_mul_add_zero _ _ ha, testBit_two_mul_add_zero _ _ hb,
      Nat.testBit_zero, mod_two_two_mul_add]
    have ha01 : a₀ = 0 ∨ a₀ = 1 := by omega
    have hb01 : b₀ = 0 ∨ b₀ = 1 := by omega
    rcases ha01 with ha01 | ha01 <;> rcases hb01 with hb01 | hb01 <;>
      rw [ha01, hb01] <;> rfl
  · have hsub : a₀ &&& b₀ ≤ 1 := by
      have hlt := Nat.and_lt_two_pow a₀ (y := b₀) (n := 1) (by rw [pow_one]; omega)
      omega
    rw [Nat.testBit_and, testBit_two_mul_add_succ _ _ _ ha, testBit_two_mul_add_succ _ _ _ hb,
      testBit_two_mul_add_succ _ _ _ hsub, Nat.testBit_and]

/-- **One bit of the carry identity**: `a₀ + b₀ = (a₀ ^^^ b₀) + 2 * (a₀ &&& b₀)` for `a₀ b₀ ≤ 1`.
This is the four-case base of the induction; the successor step is `xor_two_mul_add` and
`and_two_mul_add`. -/
theorem add_eq_xor_add_and_mul_two_of_bits {a₀ b₀ : ℕ} (ha : a₀ ≤ 1) (hb : b₀ ≤ 1) :
    a₀ + b₀ = (a₀ ^^^ b₀) + 2 * (a₀ &&& b₀) := by
  have ha01 : a₀ = 0 ∨ a₀ = 1 := by omega
  have hb01 : b₀ = 0 ∨ b₀ = 1 := by omega
  rcases ha01 with ha01 | ha01 <;> rcases hb01 with hb01 | hb01 <;>
    rw [ha01, hb01] <;> decide

/-- **The natural-number carry identity, at every pair**: `a + b = (a ^^^ b) + 2 * (a &&& b)`,
proved by strong induction on `a` with `b` carried, splitting `a = 2A + (a % 2)` and
`b = 2B + (b % 2)` and using the successor step. -/
theorem add_eq_xor_add_and_mul_two (a : ℕ) : ∀ b : ℕ, a + b = (a ^^^ b) + 2 * (a &&& b) := by
  induction a using Nat.strong_induction_on with
  | h a ih =>
    intro b
    by_cases ha : a = 0
    · rw [ha]
      simp
    have hA : a / 2 < a := by omega
    have ha₀ : a % 2 ≤ 1 := by omega
    have hb₀ : b % 2 ≤ 1 := by omega
    rw [show a = 2 * (a / 2) + a % 2 by omega, show b = 2 * (b / 2) + b % 2 by omega,
      xor_two_mul_add _ _ _ _ ha₀ hb₀, and_two_mul_add _ _ _ _ ha₀ hb₀]
    have hbits := add_eq_xor_add_and_mul_two_of_bits ha₀ hb₀
    have hih := ih (a / 2) hA (b / 2)
    omega

/-! ## 3 · The width-parametric carry identity

Everything below is quantified over the width `k`, the face `σ : ZMod (2^k)` and the pair of words.
The width-eight `decide` exclusion of the per-width file does not arise: there is no enumeration. -/

/-- **The carry identity in `ZMod (2^k)`, at every width**: the deviation of wrapping addition from
exclusive-or addition is twice the bitwise conjunction. -/
theorem denoteBV_carry (k : ℕ) (x y : BitVec k) :
    denoteBV (x + y) = denoteBV (x ^^^ y) + 2 * denoteBV (x &&& y) := by
  calc denoteBV (x + y) = ((x.toNat + y.toNat : ℕ) : ZMod (2 ^ k)) := by
        rw [denoteBV_add, denoteBV, denoteBV, Nat.cast_add]
    _ = (((x.toNat ^^^ y.toNat) + 2 * (x.toNat &&& y.toNat) : ℕ) : ZMod (2 ^ k)) := by
      rw [add_eq_xor_add_and_mul_two]
    _ = ((x.toNat ^^^ y.toNat : ℕ) : ZMod (2 ^ k))
        + 2 * ((x.toNat &&& y.toNat : ℕ) : ZMod (2 ^ k)) := by
      rw [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]

/-- Reducing the second summand modulo `n` before the outer reduction is a no-op. -/
theorem add_mod_mod_self (a b n : ℕ) : (a + b % n) % n = (a + b) % n := by
  rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]

/-- **The carry identity on `k`-bit words, at every width**: `x + y = (x ^^^ y) + 2 * (x &&& y)`
read in the word arithmetic, with the carry term shifted left by one. -/
theorem add_eq_xor_add_and_shiftLeft (k : ℕ) (x y : BitVec k) :
    x + y = (x ^^^ y) + ((x &&& y) <<< 1) := by
  refine BitVec.eq_of_toNat_eq ?_
  have hxor : (x ^^^ y).toNat = x.toNat ^^^ y.toNat := BitVec.toNat_xor _ _
  have hand : (x &&& y).toNat = x.toNat &&& y.toNat := BitVec.toNat_and _ _
  have hsh : ((x &&& y) <<< 1).toNat = ((x.toNat &&& y.toNat) * 2) % 2 ^ k := by
    rw [BitVec.toNat_shiftLeft, Nat.shiftLeft_eq, Nat.pow_one, hand]
  have hadd : ((x ^^^ y) + ((x &&& y) <<< 1)).toNat
      = ((x.toNat ^^^ y.toNat) + ((x.toNat &&& y.toNat) * 2)) % 2 ^ k := by
    rw [BitVec.toNat_add, hxor, hsh, add_mod_mod_self]
  rw [BitVec.toNat_add, hadd]
  have h2 : (x.toNat ^^^ y.toNat) + ((x.toNat &&& y.toNat) * 2) = x.toNat + y.toNat := by
    have h := add_eq_xor_add_and_mul_two x.toNat y.toNat
    omega
  rw [h2]

/-- The carry cell of `x + y`: first slot the `F2`-linear part `x ^^^ y`, second slot the
carry-generating residual `2 * (x &&& y)`. The face parameter is not used: the deviation is
face-free, since `sigma` enters the first slot only and this cell contributes `0` there. -/
def carryCell {k : ℕ} (_σ : ZMod (2 ^ k)) (x y : BitVec k) : RingCell (ZMod (2 ^ k)) :=
  (denoteBV (x ^^^ y), 2 * denoteBV (x &&& y))

/-- **The carry cell is the named pair, at every width and every face.** -/
theorem carryCell_eq_pair {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    carryCell σ x y = (denoteBV (x ^^^ y), 2 * denoteBV (x &&& y)) := rfl

/-- **The first slot of the carry cell is the exclusive-or part.** -/
theorem carryCell_fst {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).1 = denoteBV (x ^^^ y) := rfl

/-- **The second slot of the carry cell is the residual `2 * (x &&& y)`.** -/
theorem carryCell_snd {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).2 = 2 * denoteBV (x &&& y) := rfl

/-- **A product of a first-slot element and a second-slot element lands in the second slot, for
every `sigma`**: the face enters the first slot only, so the mechanism carrying the deviation is
`sigma`-independent. -/
theorem ringMul_second_of_first (R : Type*) [CommRing R] (σ a b : R) :
    ringMul σ (a, 0) (0, b) = (0, a * b) := by
  simp [ringMul]

/-- **The decomposition of `x + y` in Cell coordinates, at every width and every face**: the
`F2`-linear part as one `ringMul` against the unit, the residual as one `ringMul` of the first and
second axes. -/
theorem carryCell_eq_ringMul_sum {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    carryCell σ x y
      = ringMul σ (denoteBV (x ^^^ y), 0) ringOne
        + ringMul σ (2, 0) (0, denoteBV (x &&& y)) := by
  simp only [carryCell, ringMul]
  ext <;> simp

/-- **The two slots of the carry cell recompose the modular sum, at every width and every face.**
This is the statement the per-width file could only decide at `k = 2, 4`. -/
theorem carryCell_slots_add_sum {k : ℕ} (σ : ZMod (2 ^ k)) (x y : BitVec k) :
    (carryCell σ x y).1 + (carryCell σ x y).2 = denoteBV (x + y) := by
  simp only [carryCell]
  exact (denoteBV_carry k x y).symm

/-- Width two, as an instance of the general theorem: the slots recompose the sum in `ZMod 4`. -/
theorem residual_recomposes_sum_two : ∀ (x y : BitVec 2),
    (carryCell (0 : ZMod 4) x y).1 + (carryCell (0 : ZMod 4) x y).2 = denoteBV (x + y) :=
  fun _ _ => carryCell_slots_add_sum _ _ _

/-- Width four, as an instance of the general theorem: the slots recompose the sum in `ZMod 16`. -/
theorem residual_recomposes_sum_four : ∀ (x y : BitVec 4),
    (carryCell (0 : ZMod 16) x y).1 + (carryCell (0 : ZMod 16) x y).2 = denoteBV (x + y) :=
  fun _ _ => carryCell_slots_add_sum _ _ _

/-- Width eight, as an instance of the general theorem: the slots recompose the sum in `ZMod 256`.
The per-width file could only offer a single `255 + 255` witness here, its `2^16`-pair enumeration
being outside the lane memory envelope; this instance is `proved`, not sampled. -/
theorem residual_recomposes_sum_eight : ∀ (x y : BitVec 8),
    (carryCell (0 : ZMod 256) x y).1 + (carryCell (0 : ZMod 256) x y).2 = denoteBV (x + y) :=
  fun _ _ => carryCell_slots_add_sum _ _ _

/-- **The named residual, read in the ring**: a product of two first-slot elements, which at every
face occupies the second slot of the two-slot algebra. -/
theorem deviation_in_second_slot {k : ℕ} (σ a : ZMod (2 ^ k)) (b : ZMod (2 ^ k)) :
    ringMul σ (a, 0) (0, b) = (0, a * b) :=
  ringMul_second_of_first _ σ a b

/-! ## 4 · The algebraic normal form of bit `j` of `x + y` over `F2`

`AddPoly k` is the squarefree polynomial ring over `ZMod 2` in the `2k` input bits: `(i, false)` is
bit `i` of the first word and `(i, true)` is bit `i` of the second. Every element of this ring is
squarefree in its variables, so an element *is* its own algebraic normal form: `coeff m p` is the
ANF coefficient of the monomial `m`, and `totalDegree p` is the degree of that ANF. -/

/-- The squarefree polynomial ring over `F2` in the `2k` input bits. -/
abbrev AddPoly (k : ℕ) := MvPolynomial (Fin k × Bool) (ZMod 2)

/-- Bit `i` of the first word, as a variable of `AddPoly k`. -/
noncomputable def xVar (k : ℕ) (i : Fin k) : AddPoly k :=
  MvPolynomial.monomial (Finsupp.single (i, false) 1) 1

/-- Bit `i` of the second word, as a variable of `AddPoly k`. -/
noncomputable def yVar (k : ℕ) (i : Fin k) : AddPoly k :=
  MvPolynomial.monomial (Finsupp.single (i, true) 1) 1

/-- **The carry into bit `j`, as a polynomial.** The carry into bit `0` is zero, and the carry into
bit `j + 1` is the carry into bit `j` multiplied by the sum of the two input bits, plus the
conjunction of the two input bits. Over `F2` that is the full-adder relation
`c' = (x + y) * c + x * y`, i.e. majority in `x`, `y`, `c`. Each bit re-expands the previous carry
into a product, and that multiplication is what raises the degree. -/
noncomputable def carryPoly (k : ℕ) : Fin k → AddPoly k
  | ⟨0, _⟩ => 0
  | ⟨j + 1, h⟩ =>
      (xVar k ⟨j, Nat.lt_of_succ_lt h⟩ + yVar k ⟨j, Nat.lt_of_succ_lt h⟩)
          * carryPoly k ⟨j, Nat.lt_of_succ_lt h⟩
        + xVar k ⟨j, Nat.lt_of_succ_lt h⟩ * yVar k ⟨j, Nat.lt_of_succ_lt h⟩

/-- **Output bit `j` of `x + y`, as a polynomial**: the two input bits plus the carry into it. -/
noncomputable def bitPoly (k : ℕ) (j : Fin k) : AddPoly k := xVar k j + yVar k j + carryPoly k j

theorem carryPoly_zero (k : ℕ) (h : 0 < k) : carryPoly k ⟨0, h⟩ = 0 := by
  simp [carryPoly]

theorem carryPoly_succ (k : ℕ) (j : ℕ) (hj : j + 1 < k) :
    carryPoly k ⟨j + 1, hj⟩
      = (xVar k ⟨j, Nat.lt_of_succ_lt hj⟩ + yVar k ⟨j, Nat.lt_of_succ_lt hj⟩)
          * carryPoly k ⟨j, Nat.lt_of_succ_lt hj⟩
        + xVar k ⟨j, Nat.lt_of_succ_lt hj⟩ * yVar k ⟨j, Nat.lt_of_succ_lt hj⟩ := by
  simp [carryPoly]

/-! ### 4.1 · Exact degree of the carry polynomials, by induction on the bit index -/

theorem one_ne_zero_mod_two : (1 : ZMod 2) ≠ 0 := by
  refine fun h => ?_
  have hval := congrArg ZMod.val h
  simp at hval

theorem totalDegree_xVar (k : ℕ) (i : Fin k) : (xVar k i).totalDegree = 1 := by
  rw [xVar, MvPolynomial.totalDegree_monomial _ one_ne_zero]
  simp

theorem totalDegree_yVar (k : ℕ) (i : Fin k) : (yVar k i).totalDegree = 1 := by
  rw [yVar, MvPolynomial.totalDegree_monomial _ one_ne_zero]
  simp

/-- **The sum of the two input bits at position `i` has degree one.** The lower bound is the point:
a polynomial of total degree zero over `F2` is a constant, while the coefficient of `x_i` in
`x_i + y_i` is `1`. -/
theorem totalDegree_xVar_add_yVar (k : ℕ) (i : Fin k) : (xVar k i + yVar k i).totalDegree = 1 := by
  refine Nat.le_antisymm ?_ ?_
  · calc (xVar k i + yVar k i).totalDegree
        ≤ max (xVar k i).totalDegree (yVar k i).totalDegree := MvPolynomial.totalDegree_add _ _
      _ = 1 := by rw [totalDegree_xVar, totalDegree_yVar, max_self]
  · apply Nat.pos_of_ne_zero
    intro hz
    have hne1 : (Finsupp.single (i, false) 1 : Fin k × Bool →₀ ℕ) ≠ Finsupp.single (i, true) 1 := by
      intro e
      have h := congrArg (fun s => s (i, false)) e
      simp at h
    have hne0 : (0 : Fin k × Bool →₀ ℕ) ≠ Finsupp.single (i, false) 1 := by
      intro e
      have h := congrArg (fun s => s (i, false)) e
      simp at h
    have hc := MvPolynomial.totalDegree_eq_zero_iff_eq_C.mp hz
    have hcoeff := congrArg (MvPolynomial.coeff (Finsupp.single (i, false) 1)) hc
    have hleft : MvPolynomial.coeff (Finsupp.single (i, false) 1) (xVar k i + yVar k i) = 1 := by
      simp [xVar, yVar, MvPolynomial.coeff_monomial, Ne.symm hne1]
    have hright : MvPolynomial.coeff (Finsupp.single (i, false) 1)
        (MvPolynomial.C (MvPolynomial.coeff 0 (xVar k i + yVar k i))) = 0 := by
      simp [MvPolynomial.coeff_C, hne0]
    rw [hleft, hright] at hcoeff
    exact one_ne_zero_mod_two hcoeff

theorem ne_zero_of_totalDegree_pos {k : ℕ} {p : AddPoly k} (h : 0 < p.totalDegree) : p ≠ 0 := by
  intro hp
  subst hp
  simp at h

theorem ne_zero_of_totalDegree_ne_zero {k : ℕ} {p : AddPoly k} (h : p.totalDegree ≠ 0) : p ≠ 0 :=
  ne_zero_of_totalDegree_pos (Nat.pos_of_ne_zero h)

/-- `xVar i * yVar i` is nonzero: the coefficient of the monomial `x_i y_i` is `1`. -/
theorem xVar_mul_yVar_ne_zero (k : ℕ) (i : Fin k) : xVar k i * yVar k i ≠ 0 := by
  intro hp
  have h := congrArg
    (MvPolynomial.coeff (Finsupp.single (i, true) 1 + Finsupp.single (i, false) 1))
    (show yVar k i * xVar k i = 0 from by rw [mul_comm, hp])
  rw [yVar, xVar, MvPolynomial.coeff_mul_monomial, MvPolynomial.coeff_monomial, if_pos rfl,
    mul_one] at h
  exact one_ne_zero_mod_two h

/-- **The carry into bit `j` has algebraic normal form of degree exactly `j + 1`**, at every width
`k > j` and every `j ≥ 1`. Induction on the bit index: the carry into bit `j + 1` is the carry into
bit `j` times the linear form `x_j + y_j`, plus the quadratic `x_j y_j`, so its degree is one more
than that of its predecessor, and no higher. -/
theorem totalDegree_carryPoly_eq {k : ℕ} {j : ℕ} (hjk : j < k) (hj : 1 ≤ j) :
    (carryPoly k ⟨j, hjk⟩).totalDegree = j + 1 := by
  revert hjk hj
  refine Nat.strong_induction_on j ?_
  intro j ih hjk hj
  obtain ⟨m, hmj, _⟩ : ∃ m, j = m + 1 ∧ m < k := ⟨j - 1, by omega, by omega⟩
  subst hmj
  rw [carryPoly_succ k m hjk]
  rcases Nat.eq_zero_or_pos m with hm0 | hm0
  · subst hm0
    rw [show carryPoly k ⟨0, Nat.lt_of_succ_lt hjk⟩ = 0 from by simp [carryPoly], mul_zero, zero_add,
      MvPolynomial.totalDegree_mul_of_isDomain
        (ne_zero_of_totalDegree_pos (by rw [totalDegree_xVar]; omega))
        (ne_zero_of_totalDegree_pos (by rw [totalDegree_yVar]; omega)),
      totalDegree_xVar, totalDegree_yVar]
  · have hprev : (carryPoly k ⟨m, Nat.lt_of_succ_lt hjk⟩).totalDegree = m + 1 :=
      ih m (by omega) (Nat.lt_of_succ_lt hjk) hm0
    have hlin : (xVar k ⟨m, Nat.lt_of_succ_lt hjk⟩ + yVar k ⟨m, Nat.lt_of_succ_lt hjk⟩).totalDegree = 1 :=
      totalDegree_xVar_add_yVar k ⟨m, Nat.lt_of_succ_lt hjk⟩
    have hlinz : (xVar k ⟨m, Nat.lt_of_succ_lt hjk⟩ + yVar k ⟨m, Nat.lt_of_succ_lt hjk⟩) ≠ 0 := by
      apply ne_zero_of_totalDegree_ne_zero
      rw [hlin]
      omega
    have hcarry : (carryPoly k ⟨m, Nat.lt_of_succ_lt hjk⟩) ≠ 0 := by
      apply ne_zero_of_totalDegree_ne_zero
      rw [hprev]
      omega
    have hsmall : (xVar k ⟨m, Nat.lt_of_succ_lt hjk⟩ * yVar k ⟨m, Nat.lt_of_succ_lt hjk⟩).totalDegree
        < ((xVar k ⟨m, Nat.lt_of_succ_lt hjk⟩ + yVar k ⟨m, Nat.lt_of_succ_lt hjk⟩)
            * carryPoly k ⟨m, Nat.lt_of_succ_lt hjk⟩).totalDegree := by
      rw [MvPolynomial.totalDegree_mul_of_isDomain
            (ne_zero_of_totalDegree_pos (by rw [totalDegree_xVar]; omega))
            (ne_zero_of_totalDegree_pos (by rw [totalDegree_yVar]; omega)),
        MvPolynomial.totalDegree_mul_of_isDomain hlinz hcarry, hlin,
        totalDegree_xVar, totalDegree_yVar, hprev]
      omega
    rw [MvPolynomial.totalDegree_add_eq_left_of_totalDegree_lt hsmall,
      MvPolynomial.totalDegree_mul_of_isDomain hlinz hcarry, hlin, hprev]
    omega

/-- **No monomial of degree above `j + 1` occurs in the carry into bit `j`**, at every width. -/
theorem totalDegree_carryPoly_le {k : ℕ} {j : ℕ} (hjk : j < k) :
    (carryPoly k ⟨j, hjk⟩).totalDegree ≤ j + 1 := by
  rcases Nat.eq_zero_or_pos j with hj | hj
  · subst hj
    rw [show carryPoly k ⟨0, hjk⟩ = 0 from by simp [carryPoly]]
    simp
  · exact (totalDegree_carryPoly_eq hjk (by omega)).le

/-- **THE DEGREE THEOREM. Output bit `j` of `x + y` has algebraic normal form of degree exactly
`j + 1`, for every width `k` and every output bit `j < k`.** The per-width file could decide this only
at `k = 2, 4`; no enumeration over pairs appears here, and §5 identifies `bitPoly k j` with the
algebraic normal form of bit `j` of the sum. -/
theorem totalDegree_bitPoly (k : ℕ) (j : Fin k) : (bitPoly k j).totalDegree = j.val + 1 := by
  rcases Nat.eq_zero_or_pos j.val with hj | hj
  · have hj' : j = ⟨0, by omega⟩ := Fin.ext hj
    rw [hj']
    have h0 : carryPoly k ⟨0, by omega⟩ = 0 := by simp [carryPoly]
    rw [bitPoly, h0, add_zero]
    exact totalDegree_xVar_add_yVar _ _
  · have hlt : (xVar k j + yVar k j).totalDegree < (carryPoly k j).totalDegree := by
      rw [totalDegree_xVar_add_yVar, totalDegree_carryPoly_eq (j := j.val) (by omega) (by omega)]
      omega
    rw [bitPoly, add_comm, MvPolynomial.totalDegree_add_eq_left_of_totalDegree_lt hlt]
    exact totalDegree_carryPoly_eq (j := j.val) (by omega) (by omega)

/-! ## 5 · `bitPoly k j` is the algebraic normal form of bit `j` of the modular sum

The polynomials of §4 are the algebraic normal form of output bit `j` exactly when they evaluate to
bit `j` of `x + y` at every pair of words. That is proved next, from the natural-number carry chain,
which is a statement about sums rather than about polynomials. -/

/-- A `Bool` read as an element of `F2`. -/
def bitVal (b : Bool) : ZMod 2 := if b then 1 else 0

/-- The value of a squarefree polynomial at the bit assignment induced by the words `x`, `y`. -/
noncomputable def evalPoly {k : ℕ} (p : AddPoly k) (x y : BitVec k) : ZMod 2 :=
  MvPolynomial.eval (fun v : Fin k × Bool => bitVal (if v.2 then y.getLsb v.1 else x.getLsb v.1)) p

theorem evalPoly_zero {k : ℕ} (x y : BitVec k) : evalPoly (0 : AddPoly k) x y = 0 := by
  simp [evalPoly]

theorem evalPoly_add {k : ℕ} (p q : AddPoly k) (x y : BitVec k) :
    evalPoly (p + q) x y = evalPoly p x y + evalPoly q x y := by
  simp [evalPoly]

theorem evalPoly_mul {k : ℕ} (p q : AddPoly k) (x y : BitVec k) :
    evalPoly (p * q) x y = evalPoly p x y * evalPoly q x y := by
  simp [evalPoly]

theorem evalPoly_xVar {k : ℕ} (i : Fin k) (x y : BitVec k) :
    evalPoly (xVar k i) x y = bitVal (x.getLsb i) := by
  simp [evalPoly, xVar, MvPolynomial.eval_monomial, Finsupp.prod_single_index, bitVal]

theorem evalPoly_yVar {k : ℕ} (i : Fin k) (x y : BitVec k) :
    evalPoly (yVar k i) x y = bitVal (y.getLsb i) := by
  simp [evalPoly, yVar, MvPolynomial.eval_monomial, Finsupp.prod_single_index, bitVal]

/-! ### 5.1 · The natural-number carry chain -/

/-- Bit `i` of a natural number, as `0` or `1`. -/
def bitN (a i : ℕ) : ℕ := if a.testBit i then 1 else 0

theorem bitN_eq_mod (a i : ℕ) : bitN a i = (a / 2 ^ i) % 2 := by
  have hlt : (a / 2 ^ i) % 2 < 2 := Nat.mod_lt _ (by simp)
  rw [bitN, Nat.testBit_eq_decide_div_mod_eq]
  by_cases h : a / 2 ^ i % 2 = 1
  · simp [h]
  · have hz : a / 2 ^ i % 2 = 0 := by omega
    simp [hz]

theorem bitN_le_one (a i : ℕ) : bitN a i ≤ 1 := by
  rw [bitN_eq_mod]
  exact Nat.le_of_lt_succ (Nat.mod_lt _ (by norm_num))

/-- **Splitting off bit `i`**: the low `i + 1` bits of `a` are bit `i`, weighted by `2^i`, followed
by the low `i` bits. -/
theorem mod_two_pow_succ (a j : ℕ) :
    a % 2 ^ (j + 1) = 2 ^ j * bitN a j + a % 2 ^ j := by
  have hsplit : 2 ^ (j + 1) = 2 * 2 ^ j := by rw [pow_succ]; ring
  have hdiv : a / 2 ^ j = 2 * (a / 2 ^ (j + 1)) + bitN a j := by
    have h := Nat.div_add_mod (a / 2 ^ j) 2
    have h' : a / 2 ^ j / 2 = a / 2 ^ (j + 1) := by
      rw [Nat.div_div_eq_div_mul, show 2 ^ j * 2 = 2 ^ (j + 1) by rw [mul_comm, ← hsplit]]
    rw [bitN_eq_mod]
    omega
  have hsmall : 2 ^ j * bitN a j + a % 2 ^ j < 2 ^ (j + 1) := by
    have hbit : bitN a j ≤ 1 := bitN_le_one a j
    have hlt : a % 2 ^ j < 2 ^ j := Nat.mod_lt _ (by positivity)
    calc 2 ^ j * bitN a j + a % 2 ^ j
        < 2 ^ j * 1 + 2 ^ j := Nat.add_lt_add_of_le_of_lt (Nat.mul_le_mul_left _ hbit) hlt
      _ = 2 ^ (j + 1) := by rw [hsplit]; ring
  calc a % 2 ^ (j + 1)
      = (2 ^ j * (a / 2 ^ j) + a % 2 ^ j) % 2 ^ (j + 1) := by
        rw [Nat.div_add_mod]
    _ = (2 ^ j * (2 * (a / 2 ^ (j + 1)) + bitN a j) + a % 2 ^ j) % 2 ^ (j + 1) := by
        rw [hdiv]
    _ = (2 ^ (j + 1) * (a / 2 ^ (j + 1)) + (2 ^ j * bitN a j + a % 2 ^ j)) % 2 ^ (j + 1) := by
        rw [hsplit]; ring
    _ = 2 ^ j * bitN a j + a % 2 ^ j := by
        rw [Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hsmall]

/-- **The carry into bit `j` of `a + b`** as a natural number: the excess of the low `j` bits of the
two addends over `2^j`. It is `0` or `1`. -/
def carryNat (a b j : ℕ) : ℕ := (a % 2 ^ j + b % 2 ^ j) / 2 ^ j

theorem carryNat_lt_two (a b j : ℕ) : carryNat a b j < 2 := by
  have ha : a % 2 ^ j < 2 ^ j := Nat.mod_lt _ (by positivity)
  have hb : b % 2 ^ j < 2 ^ j := Nat.mod_lt _ (by positivity)
  rw [carryNat, Nat.div_lt_iff_lt_mul (by positivity)]
  omega

theorem carryNat_zero (a b : ℕ) : carryNat a b 0 = 0 := by
  simp [carryNat, Nat.mod_one]

/-- **The carry recursion, in natural numbers**: the carry into bit `j + 1` is the majority of the
two input bits and the carry into bit `j`, that is `(x + y + c) / 2`. -/
theorem carryNat_succ (a b j : ℕ) :
    carryNat a b (j + 1) = (bitN a j + bitN b j + carryNat a b j) / 2 := by
  have hr : (a % 2 ^ j + b % 2 ^ j) % 2 ^ j < 2 ^ j := Nat.mod_lt _ (by positivity)
  have hsplit : a % 2 ^ j + b % 2 ^ j
      = carryNat a b j * 2 ^ j + (a % 2 ^ j + b % 2 ^ j) % 2 ^ j := by
    rw [carryNat]
    exact (Nat.mod_add_div _ (2 ^ j)).symm.trans (by ring)
  have hnum : (2 ^ j * bitN a j + a % 2 ^ j) + (2 ^ j * bitN b j + b % 2 ^ j)
      = (a % 2 ^ j + b % 2 ^ j) % 2 ^ j
        + (bitN a j + bitN b j + carryNat a b j) * 2 ^ j := by
    calc (2 ^ j * bitN a j + a % 2 ^ j) + (2 ^ j * bitN b j + b % 2 ^ j)
        = 2 ^ j * (bitN a j + bitN b j) + (a % 2 ^ j + b % 2 ^ j) := by ring
      _ = 2 ^ j * (bitN a j + bitN b j)
          + (carryNat a b j * 2 ^ j + (a % 2 ^ j + b % 2 ^ j) % 2 ^ j) := by
          conv_lhs => rw [hsplit]
      _ = (a % 2 ^ j + b % 2 ^ j) % 2 ^ j
          + (bitN a j + bitN b j + carryNat a b j) * 2 ^ j := by ring
  rw [carryNat, mod_two_pow_succ, mod_two_pow_succ, hnum, pow_succ,
    ← Nat.div_div_eq_div_mul, Nat.add_mul_div_right _ _ (by positivity),
    Nat.div_eq_of_lt hr, zero_add]

/-- **The bit of a sum**: bit `j` of `a + b` is the sum of the two input bits and the carry into
that bit, reduced modulo `2`. Together with `carryNat_succ` this identifies `carryPoly` and
`bitPoly` as the algebraic normal forms of the carry and of the output bit. -/
theorem bitN_add (a b j : ℕ) :
    bitN (a + b) j = (bitN a j + bitN b j + carryNat a b j) % 2 := by
  rw [bitN_eq_mod, bitN_eq_mod a j, bitN_eq_mod b j]
  have hr : (a % 2 ^ j + b % 2 ^ j) % 2 ^ j < 2 ^ j := Nat.mod_lt _ (by positivity)
  have hsplit : a % 2 ^ j + b % 2 ^ j
      = carryNat a b j * 2 ^ j + (a % 2 ^ j + b % 2 ^ j) % 2 ^ j := by
    rw [carryNat]
    exact (Nat.mod_add_div _ (2 ^ j)).symm.trans (by ring)
  have hnum : a + b = (a % 2 ^ j + b % 2 ^ j) % 2 ^ j
      + (a / 2 ^ j + b / 2 ^ j + carryNat a b j) * 2 ^ j := by
    calc a + b
        = (2 ^ j * (a / 2 ^ j) + a % 2 ^ j) + (2 ^ j * (b / 2 ^ j) + b % 2 ^ j) := by
          rw [Nat.div_add_mod a (2 ^ j), Nat.div_add_mod b (2 ^ j)]
      _ = 2 ^ j * (a / 2 ^ j + b / 2 ^ j) + (a % 2 ^ j + b % 2 ^ j) := by ring
      _ = 2 ^ j * (a / 2 ^ j + b / 2 ^ j)
          + (carryNat a b j * 2 ^ j + (a % 2 ^ j + b % 2 ^ j) % 2 ^ j) := by
          conv_lhs => rw [hsplit]
      _ = (a % 2 ^ j + b % 2 ^ j) % 2 ^ j
          + (a / 2 ^ j + b / 2 ^ j + carryNat a b j) * 2 ^ j := by ring
  rw [hnum, Nat.add_mul_div_right _ _ (by positivity), Nat.div_eq_of_lt hr, zero_add]
  omega

theorem bitVal_testBit (a i : ℕ) : bitVal (a.testBit i) = (((a / 2 ^ i) % 2 : ℕ) : ZMod 2) := by
  rw [Nat.testBit_eq_decide_div_mod_eq]
  by_cases h : a / 2 ^ i % 2 = 1
  · simp [bitVal, h]
  · have hz : a / 2 ^ i % 2 = 0 := by omega
    simp [bitVal, hz]

theorem bitVal_getLsb (k : ℕ) (x : BitVec k) (i : Fin k) :
    bitVal (x.getLsb i) = (bitN x.toNat i.val : ZMod 2) := by
  show bitVal (x.toNat.testBit i.val) = _
  rw [bitVal_testBit, bitN_eq_mod]

/-- **The full-adder relation in `F2`**: for `x`, `y`, `c` in `{0, 1}`, the polynomial
`(x + y) * c + x * y` is the carry out of that position, `(x + y + c) / 2`. -/
theorem carry_step_mod_two (p q c : ℕ) (hp : p ≤ 1) (hq : q ≤ 1) (hc : c ≤ 1) :
    ((p : ZMod 2) + (q : ZMod 2)) * (c : ZMod 2) + (p : ZMod 2) * (q : ZMod 2)
      = (((p + q + c) / 2 : ℕ) : ZMod 2) := by
  have hcast : ((p : ZMod 2) + (q : ZMod 2)) * (c : ZMod 2) + (p : ZMod 2) * (q : ZMod 2)
      = ((((p + q) * c + p * q : ℕ)) : ZMod 2) := by
    push_cast
    ring
  rw [hcast, ZMod.natCast_eq_natCast_iff']
  have hp01 : p = 0 ∨ p = 1 := by omega
  have hq01 : q = 0 ∨ q = 1 := by omega
  have hc01 : c = 0 ∨ c = 1 := by omega
  rcases hp01 with rfl | rfl <;> rcases hq01 with rfl | rfl <;> rcases hc01 with rfl | rfl <;> decide

/-- The sum of the two input bits and the carry, in `F2`, is the corresponding bit of the sum. -/
theorem sum_three_mod_two (p q c : ℕ) (hp : p ≤ 1) (hq : q ≤ 1) (hc : c ≤ 1) :
    (p : ZMod 2) + (q : ZMod 2) + (c : ZMod 2) = (((p + q + c) % 2 : ℕ) : ZMod 2) := by
  have hle : p + q + c ≤ 3 := by omega
  have hcast : (p : ZMod 2) + (q : ZMod 2) + (c : ZMod 2) = ((p + q + c : ℕ) : ZMod 2) := by
    push_cast
    ring
  rw [hcast, ZMod.natCast_eq_natCast_iff', Nat.mod_mod]

/-! ### 5.2 · The carry polynomials and the output bits evaluate to the bits of the sum -/

/-- **The carry polynomial of bit `j` evaluates to the carry into bit `j`**, at every width. -/
theorem evalPoly_carryPoly_of_lt (k : ℕ) {j : ℕ} (hjk : j < k) (x y : BitVec k) :
    evalPoly (carryPoly k ⟨j, hjk⟩) x y = (carryNat x.toNat y.toNat j : ZMod 2) := by
  revert hjk
  refine Nat.strong_induction_on j ?_
  intro j ih hjk
  rcases Nat.eq_zero_or_pos j with hj | hj
  · subst hj
    have h0 : carryPoly k ⟨0, hjk⟩ = 0 := by simp [carryPoly]
    simp [h0, evalPoly_zero, carryNat_zero]
  · obtain ⟨m, hmj, hmK⟩ : ∃ m, j = m + 1 ∧ m < k := ⟨j - 1, by omega, by omega⟩
    subst hmj
    simp only [carryPoly_succ k m hjk, evalPoly_add, evalPoly_mul, evalPoly_xVar, evalPoly_yVar,
      ih m (by omega) (Nat.lt_of_succ_lt hjk), bitVal_getLsb, carryNat_succ]
    exact carry_step_mod_two _ _ _ (bitN_le_one _ _) (bitN_le_one _ _)
      (Nat.le_of_lt_succ (carryNat_lt_two _ _ _))

theorem evalPoly_carryPoly (k : ℕ) (j : Fin k) (x y : BitVec k) :
    evalPoly (carryPoly k j) x y = (carryNat x.toNat y.toNat j.val : ZMod 2) :=
  evalPoly_carryPoly_of_lt k j.isLt x y

/-- **`bitPoly k j` is the algebraic normal form of bit `j` of `x + y`**, in the natural-number
formulation of bits. -/
theorem evalPoly_bitPoly (k : ℕ) (j : Fin k) (x y : BitVec k) :
    evalPoly (bitPoly k j) x y = (bitN (x.toNat + y.toNat) j.val : ZMod 2) := by
  rw [bitPoly, evalPoly_add, evalPoly_add, evalPoly_xVar, evalPoly_yVar, evalPoly_carryPoly,
    bitVal_getLsb, bitVal_getLsb, ← Nat.cast_add, ← Nat.cast_add, bitN_add, Nat.cast_add,
    Nat.cast_add]
  exact sum_three_mod_two _ _ _ (bitN_le_one _ _) (bitN_le_one _ _)
    (Nat.le_of_lt_succ (carryNat_lt_two _ _ _))

theorem evalPoly_bitPoly_div (k : ℕ) (j : Fin k) (x y : BitVec k) :
    evalPoly (bitPoly k j) x y = (((x.toNat + y.toNat) / 2 ^ j.val) % 2 : ℕ) := by
  rw [evalPoly_bitPoly, bitN_eq_mod]

/-- Reducing modulo `2^k` does not change the bits below `k`. -/
theorem mod_two_pow_div_lt {A k j : ℕ} (hjk : j < k) :
    (A % 2 ^ k) / 2 ^ j % 2 = A / 2 ^ j % 2 := by
  obtain ⟨Q, R, hQR, hR⟩ : ∃ Q R, A = 2 ^ k * Q + R ∧ R < 2 ^ k :=
    ⟨A / 2 ^ k, A % 2 ^ k, (Nat.div_add_mod A (2 ^ k)).symm, Nat.mod_lt _ (by positivity)⟩
  have hk : 2 ^ k = 2 ^ j * 2 ^ (k - j) := by
    conv_lhs => rw [show k = j + (k - j) by omega]
    rw [pow_add]
  have hmod : A % 2 ^ k = R := by
    rw [hQR, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hR]
  have hdiv : A / 2 ^ j = 2 ^ (k - j) * Q + R / 2 ^ j := by
    rw [hQR, hk,
      show 2 ^ j * 2 ^ (k - j) * Q + R = R + 2 ^ j * (2 ^ (k - j) * Q) by ring,
      Nat.add_mul_div_left _ _ (by positivity)]
    ring
  have htwo : 2 ^ (k - j) * Q = 2 * (2 ^ (k - j - 1) * Q) := by
    have hkj : k - j = (k - j - 1) + 1 := by omega
    conv_lhs => rw [hkj, pow_succ]
    ring
  rw [hmod, hdiv, htwo, Nat.mul_add_mod_self_left]

/-- **`bitPoly k j` is the algebraic normal form of bit `j` of `x + y` in the word form.** Since
`AddPoly k` is squarefree, the polynomial is its own algebraic normal form, so
`totalDegree_bitPoly` reads: the `F2`-polynomial degree, in the bits of `x` and `y`, of bit `j` of
`x + y` is exactly `j + 1`, at every width and every output bit. -/
theorem evalPoly_bitPoly_getLsb (k : ℕ) (j : Fin k) (x y : BitVec k) :
    evalPoly (bitPoly k j) x y = bitVal ((x + y).getLsb j) := by
  rw [evalPoly_bitPoly_div, BitVec.getLsb, bitVal_testBit, BitVec.toNat_add]
  rw [mod_two_pow_div_lt (k := k) (j := j.val) j.isLt]

/-- The degree theorem and its identification, as one conjunction. -/
theorem bitPoly_degree_getLsb (k : ℕ) (j : Fin k) :
    (bitPoly k j).totalDegree = j.val + 1 ∧
      (∀ x y : BitVec k, evalPoly (bitPoly k j) x y = bitVal ((x + y).getLsb j)) :=
  ⟨totalDegree_bitPoly k j, fun _ _ => evalPoly_bitPoly_getLsb k j _ _⟩

/-! ## 6 · Axiom cone

Every headline above is checked in the closed cone `{propext, Classical.choice, Quot.sound}` with
no `sorryAx`. -/

#print axioms denoteBV_add
#print axioms mod_two_two_mul_add
#print axioms testBit_two_mul_add_zero
#print axioms div_two_mul_add
#print axioms testBit_two_mul_add_succ
#print axioms xor_two_mul_add
#print axioms and_two_mul_add
#print axioms add_eq_xor_add_and_mul_two_of_bits
#print axioms add_eq_xor_add_and_mul_two
#print axioms denoteBV_carry
#print axioms add_eq_xor_add_and_shiftLeft
#print axioms carryCell_eq_pair
#print axioms carryCell_fst
#print axioms carryCell_snd
#print axioms ringMul_second_of_first
#print axioms carryCell_eq_ringMul_sum
#print axioms carryCell_slots_add_sum
#print axioms residual_recomposes_sum_two
#print axioms residual_recomposes_sum_four
#print axioms residual_recomposes_sum_eight
#print axioms deviation_in_second_slot
#print axioms totalDegree_xVar_add_yVar
#print axioms xVar_mul_yVar_ne_zero
#print axioms totalDegree_carryPoly_eq
#print axioms totalDegree_carryPoly_le
#print axioms totalDegree_bitPoly
#print axioms mod_two_pow_succ
#print axioms carryNat_succ
#print axioms carryNat_lt_two
#print axioms bitN_add
#print axioms evalPoly_carryPoly
#print axioms evalPoly_bitPoly
#print axioms evalPoly_bitPoly_div
#print axioms evalPoly_bitPoly_getLsb
#print axioms bitPoly_degree_getLsb
#print axioms mod_two_pow_div_lt

end AlgebraCarryGeneral
