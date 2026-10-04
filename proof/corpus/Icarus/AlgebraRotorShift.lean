import Mathlib.Data.ZMod.Basic
import Nous.Model.EllipticModes
import Mathlib.Tactic

/-!
# The cyclic shift of an `n`-word is diagonalised by the Fourier modes, with norm-one rotors as
# eigenvalues

Task AR-24 of the algebra-rooting wave. Everything below is a theorem about the canonical objects
`Apeiron.Trichotomy.AnyRing.RingCell R` at the elliptic face `σ = -1` (the coordinate ring of
`R[x]/(x² + 1)`) and `Nous.Model.EllipticModes`, both read from the pinned corpus. Nothing is
extracted from Rust and no row refines a Rust body.

## The claim

A cyclic shift of an `n`-word is diagonalised by the discrete Fourier transform, and its eigenvalues
are `n`-th roots of unity, which are exactly the norm-one rotors `T = {z | ringNormForm (-1) z = 1}`
of the elliptic face. So word rotation is the elliptic torus action **in the spectral encoding**,
not as a ring identity on the word.

## The carrier, stated exactly

Fix a commutative ring `R`; `A := RingCell R` with `rotorMul = ringMul (-1)` is `R[x]/(x² + 1)`, and
`ringNormForm (-1) (re, im) = re² + im²`, so its norm-one locus is the unit circle `T` of
`Nous/Model/RopeOrbit.lean` over a general commutative ring instead of `ℝ`. A **primitive `n`-th root
of unity** `ζ : A` is the triple

* `hζn : rotorPow ζ n = ringOne` — `ζ` satisfies `ζ^n = 1`;
* `hprim : ∀ t < n, 0 < t → rotorPow ζ t ≠ ringOne` — no smaller positive power is `1`;
* `hunit : ringNormForm (-1) ζ = 1` — `ζ = (re, im)` with `re² + im² = 1`, i.e. `ζ` lies on `T`.

`hunit` is independent of the first two (`(1, 2) : RingCell (ZMod 5)` has norm `0`), and is carried
only where the corpus's norm machinery is used.

The function space is `Fin n → A`, the group ring `A[X]/(X^n − 1)` in its coefficient reading, with
pointwise `rotorMul` as the `A`-action. Instantiating carriers, none invented:

| `n` | `R` | `ζ` | why |
|---|---|---|---|
| 4 | `ZMod 5` | `(0, 1)` = `quarterTurnRotor` | `ζ² = (−1,0)`, `ζ⁴ = ringOne` |
| 8 | `ZMod 17` | `rotor 3 3` | `ζ² = (0,1)`, `ζ⁴ = (−1,0)`, `ζ⁸ = ringOne`; `9 + 9 = 1` |

## Trust boundary

Statements about canonical algebra over abstract commutative rings and about two concrete small
carriers. They do not reach `rustc`/LLVM, the kernel ABI, GPU/SPIR-V, the `unsafe` container, or
any measured cost. No cryptographic property is claimed or implied.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
-- Port (W2-T3): the `hn`/`h1n`/`hprim` binders below are carried in the statements (fixed
-- signatures, `rotorPow`-order families) and are deliberately not re-cited in every body.
set_option linter.unusedVariables false

namespace Nous.Algebra.RotorShift

open Apeiron.Trichotomy.AnyRing
open Nous.Model.EllipticModes

variable {R : Type*} [CommRing R]

/-! ## 1. The elliptic multiplication -/

/-- `rotorMul` is associative: it is `ringMul (-1)`, which the canonical corpus proves associative. -/
theorem rotorMul_assoc (a b c : RingCell R) :
    rotorMul (rotorMul a b) c = rotorMul a (rotorMul b c) :=
  ringMul_assoc (-1) a b c

/-- `rotorMul` is commutative. -/
theorem rotorMul_comm (a b : RingCell R) : rotorMul a b = rotorMul b a :=
  ringMul_comm (-1) a b

/-- `ringOne` is a right unit. -/
theorem rotorMul_one (a : RingCell R) : rotorMul a ringOne = a :=
  ringMul_one (-1) a

/-- `ringOne` is a left unit. -/
theorem rotorOne_mul (a : RingCell R) : rotorMul ringOne a = a :=
  ringOne_mul (-1) a

/-! ## 2. Arithmetic of reduction modulo `n` -/

/-- **One step commutes with reduction**: `((a mod n) + 1) mod n = (a + 1) mod n`, for `n ≥ 1`. -/
theorem mod_add_one_mod (a n : ℕ) (hn : 0 < n) : ((a % n) + 1) % n = (a + 1) % n := by
  rcases n with _ | m
  · omega
  · rcases m with _ | m
    · simp [Nat.mod_one]
    · have h1 : (1 : ℕ) < m + 2 := by omega
      exact ((Nat.add_mod a 1 (m + 2)).trans (by rw [Nat.mod_eq_of_lt h1])).symm

/-- **Successor past a rotation**: `(((a + b) mod n) + 1) mod n = (a + (b + 1)) mod n`. -/
theorem mod_add_mod_add_one (a b n : ℕ) (hn : 0 < n) :
    ((a + b) % n + 1) % n = (a + (b + 1)) % n :=
  mod_add_one_mod (a + b) n hn

/-- **Reduction is additive in the second argument**: `((a mod n) + b) mod n = (a + b) mod n`. -/
theorem mod_add_mod (a b n : ℕ) (hn : 0 < n) : ((a % n) + b) % n = (a + b) % n := by
  have key' : ((a % n) + b) % n = ((a % n + b) + n * (a / n)) % n :=
    (Nat.add_mul_mod_self_left (a % n + b) n (a / n)).symm
  calc ((a % n) + b) % n = ((a % n + b) + n * (a / n)) % n := key'
    _ = (a + b) % n := by rw [Nat.add_right_comm, Nat.mod_add_div]

/-! ## 3. Power laws for the rotor -/

/-- **The power is the product**: `rotorPow ζ (a + b) = ζ^a ζ^b`. -/
theorem rotorPow_add (ζ : RingCell R) (a b : ℕ) :
    rotorPow ζ (a + b) = rotorMul (rotorPow ζ a) (rotorPow ζ b) := by
  induction b with
  | zero => simp [rotorMul_one]
  | succ b ih =>
    show rotorPow ζ (a + (b + 1)) = rotorMul (rotorPow ζ a) (rotorPow ζ (b + 1))
    simp only [Nat.add_succ, rotorPow_succ, ih, rotorMul_assoc]

/-- **A multiple of the period is trivial**: if `ζ^n = 1` then `ζ^(n·q) = 1`. -/
theorem rotorPow_mul_eq_one (ζ : RingCell R) {n : ℕ} (h : rotorPow ζ n = ringOne) (q : ℕ) :
    rotorPow ζ (n * q) = ringOne := by
  induction q with
  | zero => simp
  | succ q ih =>
    show rotorPow ζ (n * (q + 1)) = ringOne
    rw [Nat.mul_succ, rotorPow_add, ih, h, rotorOne_mul]

/-- **Reduction of the exponent modulo the period**: if `ζ^n = 1` then `ζ^m = ζ^(m mod n)`. -/
theorem rotorPow_mod (ζ : RingCell R) {n m : ℕ} (h : rotorPow ζ n = ringOne) :
    rotorPow ζ m = rotorPow ζ (m % n) := by
  rw [← Nat.mod_add_div m n, rotorPow_add, rotorPow_mul_eq_one ζ h, rotorMul_one,
    Nat.add_mul_mod_self_left, Nat.mod_mod]

/-- **The power sees only the exponent modulo the period.** -/
theorem rotorPow_mod_congr (ζ : RingCell R) {n : ℕ} (h : rotorPow ζ n = ringOne) {m m' : ℕ}
    (hm : m % n = m' % n) : rotorPow ζ m = rotorPow ζ m' :=
  calc rotorPow ζ m = rotorPow ζ (m % n) := rotorPow_mod ζ h
    _ = rotorPow ζ (m' % n) := by rw [hm]
    _ = rotorPow ζ m' := (rotorPow_mod ζ h).symm

/-- **The order of the rotor is `n`**: `ζ^m = 1` exactly at the multiples of `n`. -/
theorem rotorPow_eq_one_iff_dvd (ζ : RingCell R) {n : ℕ} (hn : 0 < n)
    (hζn : rotorPow ζ n = ringOne)
    (hprim : ∀ t < n, 0 < t → rotorPow ζ t ≠ ringOne) (m : ℕ) :
    rotorPow ζ m = ringOne ↔ n ∣ m := by
  constructor
  · intro hm
    by_contra hnd
    have hlt : m % n < n := Nat.mod_lt _ hn
    have hsplit : m % n + n * (m / n) = m := Nat.mod_add_div m n
    rw [← hsplit, rotorPow_add, rotorPow_mul_eq_one ζ hζn, rotorMul_one] at hm
    have hrem : rotorPow ζ (m % n) = ringOne := by simpa using hm
    rcases Nat.eq_zero_or_pos (m % n) with hz | hp
    · exact (hnd (Nat.dvd_iff_mod_eq_zero.mpr (by simpa using hz))).elim
    · exact (hprim (m % n) hlt hp hrem).elim
  · rintro ⟨q, rfl⟩
    exact rotorPow_mul_eq_one ζ hζn q

/-- The order of the rotor is exactly `n`. -/
theorem rotorPow_order (ζ : RingCell R) {n : ℕ} (hn : 0 < n)
    (hζn : rotorPow ζ n = ringOne)
    (hprim : ∀ t < n, 0 < t → rotorPow ζ t ≠ ringOne) :
    rotorPow ζ n = ringOne ∧ ∀ t < n, 0 < t → rotorPow ζ t ≠ ringOne :=
  ⟨hζn, hprim⟩

/-! ## 4. The cyclic shift on `Fin n → RingCell R` -/

/-- The cyclic successor on `Fin n`. -/
def cycSucc (n : ℕ) (hn : 0 < n) (i : Fin n) : Fin n :=
  ⟨(i.val + 1) % n, Nat.mod_lt _ hn⟩

/-- `t` cyclic steps forward from `i`, by iteration of `cycSucc`. Temporal cost `t` modular
additions, spatial `O(1)`: the recursion applies `cycSucc` `t` times. -/
def cycStep (n : ℕ) (hn : 0 < n) : ℕ → Fin n → Fin n
  | 0, i => i
  | t + 1, i => cycSucc n hn (cycStep n hn t i)

/-- **The closed form of a rotation**: `cycStep t i` is the position `i + t` modulo `n`. -/
theorem cycStep_val (n : ℕ) (hn : 0 < n) (t : ℕ) (i : Fin n) :
    (cycStep n hn t i).val = (i.val + t) % n := by
  induction t with
  | zero =>
    show i.val = (i.val + 0) % n
    rw [Nat.add_zero, Nat.mod_eq_of_lt i.isLt]
  | succ t ih =>
    show ((cycStep n hn t i).val + 1) % n = (i.val + (t + 1)) % n
    rw [ih]
    exact mod_add_mod_add_one i.val t n hn

/-- **The cyclic shift** of an `n`-word of cells: the rotation of positions by one. -/
def shift (n : ℕ) (hn : 0 < n) (v : Fin n → RingCell R) : Fin n → RingCell R :=
  fun i => v (cycSucc n hn i)

/-- `t` applications of the shift. Temporal cost `t` function applications, spatial `O(1)` per
index: the recursion unfolds `t` times. -/
def shiftPow (n : ℕ) (hn : 0 < n) : ℕ → (Fin n → RingCell R) → (Fin n → RingCell R)
  | 0, v => v
  | t + 1, v => shift n hn (shiftPow n hn t v)

theorem shiftPow_zero' (n : ℕ) (hn : 0 < n) (v : Fin n → RingCell R) :
    shiftPow n hn 0 v = v := rfl

theorem shiftPow_succ' (n : ℕ) (hn : 0 < n) (t : ℕ) (v : Fin n → RingCell R) :
    shiftPow n hn (t + 1) v = shift n hn (shiftPow n hn t v) := rfl

/-- **`shiftPow_apply` — `shift^[t]` is rotation by `t` steps.** -/
theorem shiftPow_apply (n : ℕ) (hn : 0 < n) (t : ℕ) (v : Fin n → RingCell R) (i : Fin n) :
    shiftPow n hn t v i = v (cycStep n hn t i) := by
  induction t generalizing i with
  | zero => rfl
  | succ t ih =>
    show shift n hn (shiftPow n hn t v) i = v (cycStep n hn t.succ i)
    rw [show shift n hn (shiftPow n hn t v) i = shiftPow n hn t v (cycSucc n hn i) from rfl,
      ih (cycSucc n hn i)]
    congr 1
    apply Fin.ext
    have h1 : (cycStep n hn t (cycSucc n hn i)).val = ((i.val + 1) % n + t) % n := by
      rw [cycStep_val]
      rfl
    have h2 : (cycStep n hn t.succ i).val = (i.val + (t + 1)) % n := by
      rw [cycStep_val]
    rw [h1, h2, mod_add_mod (i.val + 1) t n hn]
    congr 1
    ring

/-- **The shift has period dividing `n`**: `n` applications are the identity on every word. -/
theorem shiftPow_n (n : ℕ) (hn : 0 < n) (v : Fin n → RingCell R) : shiftPow n hn n v = v := by
  funext i
  have h1 : shiftPow n hn n v i = v (cycStep n hn n i) := shiftPow_apply n hn n v i
  refine h1.trans (congrArg v ?_)
  exact Fin.ext (by
    rw [cycStep_val]
    exact (Nat.add_mod_right _ _).trans (Nat.mod_eq_of_lt i.isLt))

/-! ## 5. The Fourier modes, and the diagonalization -/

/-- **The `k`-th Fourier mode** of the length-`n` word space: position `i` carries `ζ^(k·i)`. -/
def fmode (n : ℕ) (ζ : RingCell R) (k : ℕ) (i : Fin n) : RingCell R :=
  rotorPow ζ (k * i.val)

theorem fmode_zero (n : ℕ) (hn : 0 < n) (ζ : RingCell R) (k : ℕ) :
    fmode n ζ k ⟨0, hn⟩ = ringOne := by
  simp [fmode]

/-- **`shift_fmode` — the diagonalization.** On the `k`-th mode the cyclic shift is the scalar
`ζ^k`: every position is multiplied by the same rotor. This is the statement that the DFT basis
diagonalises the shift, with eigenvalue `ζ^k` on the `k`-th mode. -/
theorem shift_fmode (n : ℕ) (hn : 0 < n) (ζ : RingCell R) (hζn : rotorPow ζ n = ringOne)
    (k : ℕ) (i : Fin n) :
    shift n hn (fmode n ζ k) i = rotorMul (rotorPow ζ k) (fmode n ζ k i) := by
  have h1 : shift n hn (fmode n ζ k) i = rotorPow ζ (k * ((i.val + 1) % n)) := rfl
  have h2 : rotorPow ζ (k * ((i.val + 1) % n)) = rotorPow ζ (k * (i.val + 1)) :=
    rotorPow_mod_congr ζ hζn (hm := Nat.mul_mod_mod k (i.val + 1) n)
  have h3 : rotorPow ζ (k * (i.val + 1)) = rotorMul (rotorPow ζ (k * i.val)) (rotorPow ζ k) := by
    rw [Nat.mul_succ, rotorPow_add]
  rw [h1, h2, h3, rotorMul_comm]
  rfl

/-- **`shiftPow_fmode` — iterating the shift is iterating the eigenvalue.** `t` applications of the
shift act on the `k`-th mode by `ζ^(k·t)`. -/
theorem shiftPow_fmode (n : ℕ) (hn : 0 < n) (ζ : RingCell R) (hζn : rotorPow ζ n = ringOne)
    (t k : ℕ) (i : Fin n) :
    shiftPow n hn t (fmode n ζ k) i = rotorMul (rotorPow ζ (k * t)) (fmode n ζ k i) := by
  induction t generalizing i with
  | zero =>
    have h1 : shiftPow n hn 0 (fmode n ζ k) i = fmode n ζ k i := rfl
    rw [h1, Nat.mul_zero, rotorPow_zero, rotorOne_mul]
  | succ t ih =>
    rw [show shiftPow n hn (t + 1) (fmode n ζ k) i
          = shift n hn (shiftPow n hn t (fmode n ζ k)) i from rfl,
        show shift n hn (shiftPow n hn t (fmode n ζ k)) i
          = shiftPow n hn t (fmode n ζ k) (cycSucc n hn i) from rfl, ih (cycSucc n hn i)]
    simp only [fmode]
    have key3 : rotorPow ζ (k * ((i.val + 1) % n)) = rotorPow ζ (k * (i.val + 1)) :=
      rotorPow_mod_congr ζ hζn (hm := Nat.mul_mod_mod k (i.val + 1) n)
    have key4 : rotorPow ζ (k * (i.val + 1))
        = rotorMul (rotorPow ζ (k * i.val)) (rotorPow ζ k) := by
      rw [Nat.mul_succ, rotorPow_add]
    have key5 : rotorPow ζ (k * (t + 1)) = rotorMul (rotorPow ζ (k * t)) (rotorPow ζ k) := by
      rw [Nat.mul_succ, rotorPow_add]
    rw [show (cycSucc n hn i).val = (i.val + 1) % n from rfl, key3, key4, key5]
    calc rotorMul (rotorPow ζ (k * t)) (rotorMul (rotorPow ζ (k * i.val)) (rotorPow ζ k))
        = rotorMul (rotorPow ζ (k * t)) (rotorMul (rotorPow ζ k) (rotorPow ζ (k * i.val))) := by
          rw [rotorMul_comm (rotorPow ζ (k * i.val)) (rotorPow ζ k)]
      _ = rotorMul (rotorMul (rotorPow ζ (k * t)) (rotorPow ζ k)) (rotorPow ζ (k * i.val)) :=
          (rotorMul_assoc _ _ _).symm

/-! ## 6. Norm-one: the eigenvalues are rotors on the unit circle -/

/-- **`rotorNorm_one`** — a point of the unit circle. If `re² + im² = 1` then the cell `(re, im)` has
elliptic norm form `1`: `ringNormForm (-1) (re, im) = re² + im² = 1`. This is the algebraic form of
the locus `T` of `Nous/Model/RopeOrbit.lean`. -/
theorem rotorNorm_one (re im : R) (h : re ^ 2 + im ^ 2 = 1) : ringNormForm (-1) (re, im) = 1 := by
  dsimp [ringNormForm]
  linear_combination h

/-- **`eigenvalue_unitary` — every eigenvalue of the shift is a norm-one rotor.** For
`ζ = (re, im)` on the unit circle, each power `ζ^k` has elliptic norm form `1`, by the corpus's
`unitary_rotor_preserves_norm`. -/
theorem eigenvalue_unitary (ζ : RingCell R) (hunit : ringNormForm (-1) ζ = 1) (k : ℕ) :
    ringNormForm (-1) (rotorPow ζ k) = 1 :=
  unitary_rotor_preserves_norm ζ hunit k

/-- **Every value of every mode is a norm-one rotor.** -/
theorem fmode_unitary (n : ℕ) (ζ : RingCell R) (hunit : ringNormForm (-1) ζ = 1) (k : ℕ)
    (i : Fin n) : ringNormForm (-1) (fmode n ζ k i) = 1 :=
  unitary_rotor_preserves_norm ζ hunit (k * i.val)

/-- **`shiftPow_eigen_unitary` — every `t`-fold shift is unitary on every mode.** The eigenvalue of
`shift^[t]` on mode `k` is `ζ^(k·t)`, whose norm form is `1`, so `t` applications of the shift are
`t` applications of a norm-one rotor. -/
theorem shiftPow_eigen_unitary (n : ℕ) (hn : 0 < n) (ζ : RingCell R) (hunit : ringNormForm (-1) ζ = 1)
    (hζn : rotorPow ζ n = ringOne) (t k : ℕ) (i : Fin n) :
    ringNormForm (-1) (rotorPow ζ (k * t)) = 1 ∧
      shiftPow n hn t (fmode n ζ k) i = rotorMul (rotorPow ζ (k * t)) (fmode n ζ k i) :=
  ⟨eigenvalue_unitary ζ hunit (k * t), shiftPow_fmode n hn ζ hζn t k i⟩

/-! ## 7. The order statement: the shift's order is the rotor's order -/

/-- **`shift_order_n` — the cyclic shift has order exactly `n`.** `n` applications are the identity
on every word, and no smaller positive number of applications is, witnessed on the mode `1`. -/
theorem shift_order_n (n : ℕ) (hn : 0 < n) (h1n : 1 < n) (ζ : RingCell R)
    (hζn : rotorPow ζ n = ringOne)
    (hprim : ∀ t < n, 0 < t → rotorPow ζ t ≠ ringOne) :
    (∀ v : Fin n → RingCell R, shiftPow n hn n v = v)
    ∧ ∀ t < n, 0 < t → shiftPow n hn t (fmode n ζ 1) ≠ fmode n ζ 1 := by
  refine ⟨fun v => shiftPow_n n hn v, ?_⟩
  intro t ht ht0 hEq
  have h1 := congrFun hEq ⟨0, hn⟩
  have h2 := shiftPow_fmode n hn ζ hζn t 1 ⟨0, hn⟩
  have h3 : fmode n ζ 1 ⟨0, hn⟩ = ringOne := fmode_zero n hn ζ 1
  rw [h3] at h1
  rw [h3, Nat.one_mul, rotorMul_one] at h2
  have hkey : rotorPow ζ t = ringOne := (h1.symm.trans h2).symm
  exact (hprim t ht ht0 hkey).elim

/-- **`period_eq_rotor_order` — the period of a mode is the order of the corresponding rotor.**
The shift fixes mode `k` after `t` steps exactly when `ζ^(k·t) = 1`, which by
`rotorPow_eq_one_iff_dvd` is exactly `n ∣ k·t`. -/
theorem period_eq_rotor_order (n : ℕ) (hn : 0 < n) (ζ : RingCell R) (hζn : rotorPow ζ n = ringOne)
    (hprim : ∀ t < n, 0 < t → rotorPow ζ t ≠ ringOne) (t k : ℕ) :
    (∀ i, shiftPow n hn t (fmode n ζ k) i = fmode n ζ k i) ↔ rotorPow ζ (k * t) = ringOne := by
  constructor
  · intro hfix
    have h0 := hfix ⟨0, hn⟩
    have h2 := shiftPow_fmode n hn ζ hζn t k ⟨0, hn⟩
    have h3 : fmode n ζ k ⟨0, hn⟩ = ringOne := fmode_zero n hn ζ k
    rw [h3] at h0
    rw [h3, rotorMul_one] at h2
    exact (h0.symm.trans h2).symm
  · intro hkt i
    rw [shiftPow_fmode n hn ζ hζn t k i, hkt, rotorOne_mul]

/-- **Mode 1 has period `n`.** The order of the shift on mode 1 is exactly the order of the rotor
`ζ`: `ζ^t = 1` iff `shift^[t]` fixes mode 1. -/
theorem period_one_eq_rotor_order (n : ℕ) (hn : 0 < n) (ζ : RingCell R) (hζn : rotorPow ζ n = ringOne)
    (hprim : ∀ t < n, 0 < t → rotorPow ζ t ≠ ringOne) (t : ℕ) :
    (∀ i, shiftPow n hn t (fmode n ζ 1) i = fmode n ζ 1 i) ↔ rotorPow ζ t = ringOne := by
  constructor
  · intro hfix
    have h0 := hfix ⟨0, hn⟩
    have h2 := shiftPow_fmode n hn ζ hζn t 1 ⟨0, hn⟩
    have h3 : fmode n ζ 1 ⟨0, hn⟩ = ringOne := fmode_zero n hn ζ 1
    rw [h3] at h0
    rw [h3, Nat.one_mul, rotorMul_one] at h2
    exact (h0.symm.trans h2).symm
  · intro ht i
    rw [shiftPow_fmode n hn ζ hζn t 1 i, Nat.one_mul, ht, rotorOne_mul]

/-! ## 8. Verdict -/

/-- **`shift_roots_onto_elliptic_face` — the verdict.** For a primitive `n`-th root of unity `ζ` on
the unit circle of the elliptic face: the cyclic shift of an `n`-word is diagonalised by the Fourier
modes, with eigenvalue `ζ^k` on mode `k`; every eigenvalue is a norm-one rotor; the shift has order
exactly `n`, which is the order of `ζ`. Word rotation is the elliptic torus action in the spectral
encoding. -/
theorem shift_roots_onto_elliptic_face (n : ℕ) (hn : 0 < n) (h1n : 1 < n) (ζ : RingCell R)
    (hζn : rotorPow ζ n = ringOne) (hunit : ringNormForm (-1) ζ = 1)
    (hprim : ∀ t < n, 0 < t → rotorPow ζ t ≠ ringOne) :
    (∀ k : ℕ, ∀ i : Fin n, shift n hn (fmode n ζ k) i = rotorMul (rotorPow ζ k) (fmode n ζ k i))
    ∧ (∀ k, ringNormForm (-1) (rotorPow ζ k) = 1)
    ∧ (∀ v : Fin n → RingCell R, shiftPow n hn n v = v)
    ∧ (∀ t < n, 0 < t → shiftPow n hn t (fmode n ζ 1) ≠ fmode n ζ 1)
    ∧ rotorPow ζ n = ringOne :=
  ⟨fun k i => shift_fmode n hn ζ hζn k i,
   fun k => eigenvalue_unitary ζ hunit k,
   fun v => shiftPow_n n hn v,
   (shift_order_n n hn h1n ζ hζn hprim).2,
   hζn⟩

/-! ## 9. Concrete instances -/

section Instances

/-- The order-4 elliptic rotor `(0, 1)` over `ZMod 5`. -/
def ζ4 : RingCell (ZMod 5) := quarterTurnRotor (R := ZMod 5)

theorem inst4_period : rotorPow ζ4 4 = ringOne := by decide
theorem inst4_eig0 : rotorPow ζ4 0 = ringOne := by decide
theorem inst4_eig1 : rotorPow ζ4 1 = (0, 1) := by decide
theorem inst4_eig2 : rotorPow ζ4 2 = (-1, 0) := by decide
theorem inst4_eig3 : rotorPow ζ4 3 = (0, -1) := by decide

theorem inst4_unitary (k : ℕ) : ringNormForm (-1) (rotorPow ζ4 k) = 1 :=
  unitary_rotor_preserves_norm ζ4 (quarterTurnRotor_unitary) k

theorem inst4_primitive (t : ℕ) (ht : t < 4) (ht0 : 0 < t) : rotorPow ζ4 t ≠ ringOne := by
  interval_cases t <;> decide

/-- **The diagonalization instantiated at `n = 4` over `ZMod 5`.** -/
theorem inst4_diag (k i : Fin 4) :
    shift 4 (by norm_num) (fmode 4 ζ4 k) i = rotorMul (rotorPow ζ4 k) (fmode 4 ζ4 k i) :=
  shift_fmode 4 (by norm_num) ζ4 inst4_period k i

/-- **The verdict instantiated at `n = 4` over `ZMod 5`.** -/
theorem inst4_verdict :
    (∀ k : ℕ, ∀ i : Fin 4,
        shift 4 (by norm_num) (fmode 4 ζ4 k) i = rotorMul (rotorPow ζ4 k) (fmode 4 ζ4 k i))
    ∧ (∀ k : ℕ, ringNormForm (-1) (rotorPow ζ4 k) = 1)
    ∧ (∀ v : Fin 4 → RingCell (ZMod 5), shiftPow 4 (by norm_num) 4 v = v)
    ∧ (∀ t < 4, 0 < t → shiftPow 4 (by norm_num) t (fmode 4 ζ4 1) ≠ fmode 4 ζ4 1)
    ∧ rotorPow ζ4 4 = ringOne := by
  have hn : (0 : ℕ) < 4 := by norm_num
  have h1n : (1 : ℕ) < 4 := by norm_num
  exact shift_roots_onto_elliptic_face 4 hn h1n ζ4 inst4_period (quarterTurnRotor_unitary)
    inst4_primitive

/-- The order-8 elliptic rotor `(3, 3)` over `ZMod 17`: `9 + 9 = 1`, `ζ² = (0,1)`, `ζ⁴ = (−1,0)`. -/
def ζ8 : RingCell (ZMod 17) := rotor 3 3

theorem inst8_period : rotorPow ζ8 8 = ringOne := by decide
theorem inst8_unitary : ringNormForm (-1) ζ8 = 1 := by decide
theorem inst8_primitive (t : ℕ) (ht : t < 8) (ht0 : 0 < t) : rotorPow ζ8 t ≠ ringOne := by
  interval_cases t <;> decide

theorem inst8_diag (k i : Fin 8) :
    shift 8 (by norm_num) (fmode 8 ζ8 k) i = rotorMul (rotorPow ζ8 k) (fmode 8 ζ8 k i) :=
  shift_fmode 8 (by norm_num) ζ8 inst8_period k i

/-- **The verdict instantiated at `n = 8` over `ZMod 17`.** -/
theorem inst8_verdict :
    (∀ k : ℕ, ∀ i : Fin 8,
        shift 8 (by norm_num) (fmode 8 ζ8 k) i = rotorMul (rotorPow ζ8 k) (fmode 8 ζ8 k i))
    ∧ (∀ k : ℕ, ringNormForm (-1) (rotorPow ζ8 k) = 1)
    ∧ (∀ v : Fin 8 → RingCell (ZMod 17), shiftPow 8 (by norm_num) 8 v = v)
    ∧ (∀ t < 8, 0 < t → shiftPow 8 (by norm_num) t (fmode 8 ζ8 1) ≠ fmode 8 ζ8 1)
    ∧ rotorPow ζ8 8 = ringOne := by
  have hn : (0 : ℕ) < 8 := by norm_num
  have h1n : (1 : ℕ) < 8 := by norm_num
  exact shift_roots_onto_elliptic_face 8 hn h1n ζ8 inst8_period inst8_unitary inst8_primitive

end Instances

/-! ## 10. Scoping against the AR-14 congruence obstruction

`Ccong j u v` is the notion of `work/proofs/AlgebraSha256d.md` §3: the two cells differ by a
multiple of `2^j`. That file lives outside the corpus, so nothing is imported from it; the notion
is restated here and proved over any commutative ring.
-/

/-- `Ccong j u v`: the two cells differ by a multiple of `2^j`, i.e. `u - v` lies in the ideal
`(2^j)` of the cell ring. -/
def Ccong (j : ℕ) (u v : RingCell R) : Prop :=
  ∃ c : RingCell R, u - v = (2 : R) ^ j • c

/-- `a • (b • x) = b • (a • x)` for the scalar action on the cell ring. The scalar action on
`R × R` is coordinatewise (`Prod.smul_fst`, `Prod.smul_snd`), and on `R` it is multiplication
(`smul_eq_mul`), so each coordinate reduces to a ring identity. -/
theorem smul_smul_swap (a b : R) (x : RingCell R) : a • (b • x) = b • (a • x) := by
  apply Prod.ext
  · simp only [Prod.smul_fst, smul_eq_mul]
    ring
  · simp only [Prod.smul_snd, smul_eq_mul]
    ring

/-- `(a + c) - (b + d) = (a - b) + (c - d)`. -/
theorem sub_add_sub' (a b c d : RingCell R) : (a + c) - (b + d) = (a - b) + (c - d) :=
  Prod.ext (by ring) (by ring)

/-- `c • (u - v) = c • u - c • v`. -/
theorem smul_sub' (c : R) (u v : RingCell R) : c • (u - v) = c • u - c • v := by
  apply Prod.ext
  · simp only [Prod.smul_fst, Prod.fst_sub, smul_eq_mul]
    ring
  · simp only [Prod.smul_snd, Prod.snd_sub, smul_eq_mul]
    ring

/-- `c • (u + v) = c • u + c • v`. -/
theorem smul_add' (c : R) (u v : RingCell R) : c • (u + v) = c • u + c • v := by
  apply Prod.ext
  · simp only [Prod.smul_fst, Prod.fst_add, smul_eq_mul]
    ring
  · simp only [Prod.smul_snd, Prod.snd_add, smul_eq_mul]
    ring

/-- `c • 0 = 0` for the scalar action on the cell ring. -/
theorem smul_zero' (c : R) : c • ((0 : RingCell R)) = (0 : RingCell R) := by
  apply Prod.ext <;> simp

/-- Congruence is reflexive. -/
theorem Ccong_refl (j : ℕ) (u : RingCell R) : Ccong j u u := by
  refine ⟨0, ?_⟩
  apply Prod.ext <;> simp

/-- Congruence is preserved by a common cell added to both sides. -/
theorem Ccong_add (j : ℕ) {a b c d : RingCell R} (hab : Ccong j a b) (hcd : Ccong j c d) :
    Ccong j (a + c) (b + d) := by
  obtain ⟨x, hx⟩ := hab
  obtain ⟨y, hy⟩ := hcd
  refine ⟨x + y, ?_⟩
  calc (a + c) - (b + d) = (a - b) + (c - d) := sub_add_sub' a b c d
    _ = (2 : R) ^ j • x + (2 : R) ^ j • y := by rw [hx, hy]
    _ = (2 : R) ^ j • (x + y) := (smul_add' _ _ _).symm

/-- Congruence is preserved by a scalar from the base ring applied to both sides. -/
theorem Ccong_smul (j : ℕ) (c : R) {u v : RingCell R} (h : Ccong j u v) :
    Ccong j (c • u) (c • v) := by
  obtain ⟨x, hx⟩ := h
  refine ⟨c • x, ?_⟩
  calc c • u - c • v = c • (u - v) := (smul_sub' c u v).symm
    _ = c • ((2 : R) ^ j • x) := by rw [hx]
    _ = (2 : R) ^ j • (c • x) := (smul_smul_swap _ _ _).symm

/-- **`Ccong_rotorMul_left` — congruence is preserved by the elliptic product in its left argument.**
This is the `σ = -1` instance of AR-14's `Ccong_mul_left` (§D item 10, discharged in that file's
run16): each coordinate of `rotorMul w u` is a signed product of one coordinate of the fixed
`w` with one coordinate of `u`, and a product of a multiple of `2^j` with anything is a multiple of
`2^j`. The witness is `rotorMul w a`. -/
theorem Ccong_rotorMul_left (j : ℕ) {u u' : RingCell R} (w : RingCell R) (h : Ccong j u u') :
    Ccong j (rotorMul w u) (rotorMul w u') := by
  obtain ⟨a, ha⟩ := h
  refine ⟨rotorMul w a, ?_⟩
  have hsub : rotorMul w u - rotorMul w u' = rotorMul w (u - u') := by
    dsimp only [rotorMul]
    exact (ringMul_distrib_sub (-1) w u u').symm
  rw [hsub, ha]
  apply Prod.ext
  · simp only [rotorMul, ringMul, Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
    ring
  · simp only [rotorMul, ringMul, Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
    ring

/-- A constant scalar pulls out of a finite sum. -/
theorem sum_smul (ι : Type*) [Fintype ι] (a : R) (c : ι → RingCell R) :
    ∑ i, a • c i = a • ∑ i, c i := by
  classical
  have key : ∀ s : Finset ι, ∑ i ∈ s, a • c i = a • ∑ i ∈ s, c i := by
    intro s
    induction s using Finset.induction_on with
    | empty => simp
    | insert i s hi ih => simp only [Finset.sum_insert hi, ih, smul_add']
  simpa using key Finset.univ

/-- Congruence is preserved by a finite sum: the spectral encoding's coordinates are sums. -/
theorem Ccong_sum {ι : Type*} [Fintype ι] (j : ℕ) {f g : ι → RingCell R}
    (h : ∀ i, Ccong j (f i) (g i)) : Ccong j (∑ i, f i) (∑ i, g i) := by
  have key : ∀ i, f i - g i = (2 : R) ^ j • Classical.choose (h i) :=
    fun i => (h i).choose_spec
  refine ⟨∑ i, Classical.choose (h i), ?_⟩
  calc (∑ i, f i) - (∑ i, g i) = ∑ i, (f i - g i) := (Finset.sum_sub_distrib (f := f) (g := g)).symm
    _ = ∑ i, (2 : R) ^ j • Classical.choose (h i) := Finset.sum_congr rfl fun i _ => key i
    _ = (2 : R) ^ j • ∑ i, Classical.choose (h i) := sum_smul _ _ _

/-- **The discrete Fourier transform of a word of cells**, at mode `k`: the `i`-th cell is multiplied
by the rotor `ζ^(k·i)`. This is the spectral encoding of the word, under the `A`-action
`rotorMul`. -/
def dft (n : ℕ) (ζ : RingCell R) (w : Fin n → RingCell R) (k : Fin n) : RingCell R :=
  ∑ i : Fin n, rotorMul (rotorPow ζ (k.val * i.val)) (w i)

/-- **`dft_cong_preserves` — the spectral encoding IS `2^j`-congruence compatible.** If two words are
coordinatewise congruent modulo `2^j`, at every `j`, their spectra are congruent modulo `2^j`. The
encoding is a sum of elliptic products, and by `Ccong_rotorMul_left` each such product carries the
ideal `(2^j)` into itself. This is the correction of the premise in the task statement: the DFT is
congruence compatible, and it is built from exactly the operations the AR-14 obstruction covers, so
the two results do not conflict. -/
theorem dft_cong_preserves (j : ℕ) (n : ℕ) (ζ : RingCell R) {w w' : Fin n → RingCell R}
    (h : ∀ i, Ccong j (w i) (w' i)) (k : Fin n) :
    Ccong j (dft n ζ w k) (dft n ζ w' k) := by
  refine Ccong_sum (R := R) j
    (f := fun i => rotorMul (rotorPow ζ (k.val * i.val)) (w i))
    (g := fun i => rotorMul (rotorPow ζ (k.val * i.val)) (w' i)) ?_
  intro i
  exact Ccong_rotorMul_left (R := R) j (rotorPow ζ (k.val * i.val)) (h i)

/-- The constant word of length 4 with every entry `(a, 0)`. -/
def w4 (a : R) : Fin 4 → RingCell R := fun _ => (a, 0)

/-- **`dft_collision_ZMod4` — the smallest concrete witness at `n = 4` that the spectral encoding is
not an encoding.** Over the word ring `ZMod (2²) = ZMod 4` with the elliptic rotor `ζ = (0,1)`, the
distinct words `(1,1,1,1)` and `(0,0,0,0)` — which are not congruent modulo `2` at position `0`,
since `1 - 0 = 1` — have the **same** spectrum, namely `0` in every mode. The spectral encoding
therefore identifies congruence classes rather than separating them: it loses the word. This is why
the diagonalization above is a statement about the operator on `Fin n → RingCell R` and not a ring
identity on the word: over a `2`-power word ring the encoding is not injective. -/
theorem dft_collision_ZMod4 :
    (∀ k : Fin 4, dft 4 (quarterTurnRotor (R := ZMod 4)) (w4 (1 : ZMod 4)) k
        = dft 4 (quarterTurnRotor (R := ZMod 4)) (w4 0) k)
    ∧ (w4 (1 : ZMod 4)) 0 ≠ (w4 0) 0 := by
  refine ⟨by decide, by decide⟩

/-- **`no_decoder_ZMod4` — the spectral route yields no map on word values.** There is no function
that inverts the length-4 spectral encoding over `ZMod (2²)`, so the spectral encoding cannot be
used to compute a shift of a word value: it does not reach the word at all. This is the precise
sense in which the AR-14 congruence obstruction over `RingCell (ZMod (2^w))` and the spectral
diagonalization do not conflict. -/
theorem no_decoder_ZMod4 :
    ¬ ∃ dec : (Fin 4 → RingCell (ZMod 4)) → (Fin 4 → RingCell (ZMod 4)),
      ∀ w : Fin 4 → RingCell (ZMod 4), dec (dft 4 (quarterTurnRotor (R := ZMod 4)) w) = w := by
  rintro ⟨dec, hdec⟩
  have h1 : dec (dft 4 (quarterTurnRotor (R := ZMod 4)) (w4 (1 : ZMod 4))) (0 : Fin 4)
      = (w4 (1 : ZMod 4)) (0 : Fin 4) := congrFun (hdec (w4 (1 : ZMod 4))) (0 : Fin 4)
  have h0 : dec (dft 4 (quarterTurnRotor (R := ZMod 4)) (w4 (0 : ZMod 4))) (0 : Fin 4)
      = (w4 (0 : ZMod 4)) (0 : Fin 4) := congrFun (hdec (w4 (0 : ZMod 4))) (0 : Fin 4)
  have hdec1 : dec (dft 4 (quarterTurnRotor (R := ZMod 4)) (w4 (1 : ZMod 4))) (0 : Fin 4)
      = dec (dft 4 (quarterTurnRotor (R := ZMod 4)) (w4 (0 : ZMod 4))) (0 : Fin 4) := by
    congr 1
    funext k
    exact dft_collision_ZMod4.1 k
  have hkey : (w4 (1 : ZMod 4)) 0 = (w4 (0 : ZMod 4)) 0 := by
    calc (w4 (1 : ZMod 4)) 0 = dec (dft 4 (quarterTurnRotor (R := ZMod 4)) (w4 (1 : ZMod 4))) 0 :=
          h1.symm
      _ = dec (dft 4 (quarterTurnRotor (R := ZMod 4)) (w4 (0 : ZMod 4))) 0 := hdec1
      _ = (w4 (0 : ZMod 4)) 0 := h0
  exact dft_collision_ZMod4.2 hkey

/-- **The collapse above is a property of the `2`-power modulus, not of the transform.** Over
`ZMod 5`, where `n = 4` is a unit, the same word has the nonzero spectrum `(4, 0)` in mode `0`. -/
theorem dft_ZMod5_value :
    dft 4 (quarterTurnRotor (R := ZMod 5)) (w4 (1 : ZMod 5)) 0 = (4, 0) := by
  decide

theorem dft_ZMod5_distinct :
    dft 4 (quarterTurnRotor (R := ZMod 5)) (w4 (1 : ZMod 5)) 0
      ≠ dft 4 (quarterTurnRotor (R := ZMod 5)) (w4 0) 0 := by
  decide

/-! ## 11. Axiom cones -/

#print axioms rotorPow_add
#print axioms rotorPow_mul_eq_one
#print axioms rotorPow_mod
#print axioms rotorPow_eq_one_iff_dvd
#print axioms shift_fmode
#print axioms shiftPow_fmode
#print axioms shiftPow_n
#print axioms rotorNorm_one
#print axioms eigenvalue_unitary
#print axioms fmode_unitary
#print axioms shiftPow_eigen_unitary
#print axioms shift_order_n
#print axioms period_eq_rotor_order
#print axioms period_one_eq_rotor_order
#print axioms shift_roots_onto_elliptic_face
#print axioms inst4_diag
#print axioms inst4_verdict
#print axioms inst8_diag
#print axioms inst8_verdict
#print axioms Ccong_rotorMul_left
#print axioms dft_cong_preserves
#print axioms dft_collision_ZMod4
#print axioms no_decoder_ZMod4
#print axioms dft_ZMod5_distinct

end Nous.Algebra.RotorShift