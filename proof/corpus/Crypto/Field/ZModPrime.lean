/-
Executable prime-field helpers over `ZMod p` for `[Fact p.Prime]`, in the part of the gap list
where the pinned Mathlib has no implementation.

`Mathlib.Algebra.Field.ZMod` gives `Field (ZMod p)` only under `[Fact p.Prime]`, and that
instance does not reduce: `(4 : ZMod 7) * (4 : ZMod 7)⁻¹ = 1` is closed by `native_decide` but
not by `decide`, because `Inv.inv` routes through `ZMod.natCast_zmod_surjective`. Every
correctness statement here therefore goes through the computable `ZMod.inv`
(`Nat.gcdA a.val p`) through `ZMod.mul_inv_eq_gcd`, or through Euler's criterion
(`ZMod.euler_criterion`), never through `Inv.inv` of the `Field` instance. No Tonelli–Shanks
inner loop and no primitive-root construction are implemented here; the Tonelli–Shanks entry
point below carries the algorithm's preconditions and its conclusion, and says which of them
the proof uses.

Everything in this file is generic in the prime `p`. The only concrete values are the
executable witnesses in the last section, which exist to run the definitions.
-/

import Mathlib.Algebra.Field.ZMod
import Mathlib.NumberTheory.LegendreSymbol.Basic

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Field

open ZMod

variable {p : ℕ} [Fact p.Prime]

/-- `p` is nonzero, from its primality. File-local: it is needed for the `ZMod.val` lemmas and
is deliberately not registered as a global `NeZero` instance. -/
local instance instNeZero : NeZero p := ⟨(Fact.out : Nat.Prime p).ne_zero⟩

/-! ## Inversion -/

/-- Executable inverse in the prime field: `ZMod.inv`, whose value is the extended-Euclid
witness `Nat.gcdA a.val p` (`Mathlib/Data/ZMod/Basic.lean`). Cost: one extended-Euclid run,
`O(log p)` `Nat` remainder steps in the recurrence `r_{k+1} = r_{k-1} mod r_k`, which closes
after `O(log p)` steps by the Fibonacci bound on consecutive remainders; `O(1)` extra `Nat`s. -/
def invZMod (a : ZMod p) : ZMod p := a⁻¹

omit [Fact p.Prime] in
theorem invZMod_eq_inv (a : ZMod p) : invZMod a = ZMod.inv p a := rfl

theorem mul_invZMod_of_ne_zero (a : ZMod p) (ha : a ≠ 0) : a * invZMod a = 1 := by
  have hval : a.val ≠ 0 := by
    intro hz
    refine ha ?_
    rw [← ZMod.natCast_zmod_val a, hz, Nat.cast_zero]
  have hcop : Nat.Coprime p a.val :=
    Nat.coprime_of_lt_prime (n := a.val) hval (ZMod.val_lt a) (Fact.out : Nat.Prime p)
  calc a * invZMod a = Nat.gcd a.val p := ZMod.mul_inv_eq_gcd a
    _ = 1 := by rw [Nat.coprime_iff_gcd_eq_one.mp hcop.symm]; norm_num

theorem invZMod_mul_of_ne_zero (a : ZMod p) (ha : a ≠ 0) : invZMod a * a = 1 := by
  rw [mul_comm, mul_invZMod_of_ne_zero a ha]

/-- Fermat: in a prime field the inverse of a nonzero element is `a ^ (p - 2)`. This exponent
costs `O(log p)` multiplications in `ZMod p` by repeated squaring, against the `O(log p)`
`Nat` remainder steps of `invZMod`. -/
theorem invZMod_eq_pow_sub_two (a : ZMod p) (ha : a ≠ 0) : invZMod a = a ^ (p - 2) := by
  have hpos : 2 ≤ p := (Fact.out : Nat.Prime p).two_le
  have hsub : p - 2 + 1 = p - 1 := by omega
  refine mul_left_cancel₀ ha ?_
  calc a * invZMod a = 1 := mul_invZMod_of_ne_zero a ha
    _ = a * a ^ (p - 2) := by
        rw [mul_comm, ← pow_succ, hsub]
        exact (ZMod.pow_card_sub_one_eq_one ha).symm

/-! ## Exponentiation -/

/-- Executable exponentiation by structural recursion on the exponent. Recurrence
`f a 0 = 1`, `f a (e + 1) = a * f a e`: `e` multiplications in `ZMod p`; `O(1)` `ZMod p`
values live at any level of the recursion and the stack depth is the exponent, so `O(e)`
words if the recursion is not turned into a loop. The recursion closes because its measure is
the exponent, which decreases by one per step. -/
def powZMod (a : ZMod p) : ℕ → ZMod p
  | 0 => 1
  | e + 1 => a * powZMod a e

theorem powZMod_zero (a : ZMod p) : powZMod a 0 = 1 := rfl

theorem powZMod_succ (a : ZMod p) (e : ℕ) : powZMod a (e + 1) = a * powZMod a e := rfl

theorem powZMod_eq (a : ZMod p) (e : ℕ) : powZMod a e = a ^ e := by
  induction e with
  | zero => exact (powZMod_zero a).trans (Monoid.npow_zero a).symm
  | succ e ih =>
      exact (powZMod_succ a e).trans (congrArg (fun z => a * z) ih) |>.trans (pow_succ' a e).symm

/-! ## The Legendre symbol -/

/-- The Legendre symbol of `a` at the prime `p`: the quadratic character `quadraticChar` of
`ZMod p`, which is the body of `legendreSym p` (the pinned
`Mathlib/NumberTheory/LegendreSymbol/Basic.lean`) with the integer lift of `a` replaced by `a`.
Cost: `quadraticChar` is a table lookup over the `Fintype (ZMod p)` structure, so the current
definition materialises `p` entries per evaluation: `O(p)` `ZMod p` comparisons, `O(p)` words. -/
def legendre (a : ZMod p) : ℤ := quadraticChar (ZMod p) a

@[simp] theorem legendre_zero : legendre (0 : ZMod p) = 0 := by
  show quadraticChar (ZMod p) (0 : ZMod p) = 0
  exact quadraticChar_zero

@[simp] theorem legendre_one : legendre (1 : ZMod p) = 1 := by
  show quadraticChar (ZMod p) (1 : ZMod p) = 1
  exact quadraticCharFun_one

theorem legendre_mul (a b : ZMod p) : legendre (a * b) = legendre a * legendre b := by
  show quadraticCharFun (ZMod p) (a * b)
      = quadraticCharFun (ZMod p) a * quadraticCharFun (ZMod p) b
  exact quadraticCharFun_mul a b

theorem legendre_eq_zero_iff (a : ZMod p) : legendre a = 0 ↔ a = 0 := by
  show quadraticChar (ZMod p) a = 0 ↔ a = 0
  exact quadraticChar_eq_zero_iff

theorem legendre_eq_neg_one_iff (a : ZMod p) : legendre a = -1 ↔ ¬ IsSquare a := by
  show quadraticChar (ZMod p) a = -1 ↔ ¬ IsSquare a
  exact quadraticChar_neg_one_iff_not_isSquare

theorem legendre_sq_one (a : ZMod p) (ha : a ≠ 0) : legendre a ^ 2 = 1 := by
  show quadraticChar (ZMod p) a ^ 2 = 1
  exact quadraticChar_sq_one ha

theorem legendre_eq_one_or_neg_one (a : ZMod p) (ha : a ≠ 0) :
    legendre a = 1 ∨ legendre a = -1 := by
  show quadraticChar (ZMod p) a = 1 ∨ quadraticChar (ZMod p) a = -1
  exact quadraticChar_dichotomy ha

/-- For a nonzero element the symbol is `1` exactly on the squares: the decision procedure for
squareness that does not go through the existential in `IsSquare`. -/
theorem legendre_eq_one_iff (a : ZMod p) (ha : a ≠ 0) : legendre a = 1 ↔ IsSquare a := by
  show quadraticChar (ZMod p) a = 1 ↔ IsSquare a
  exact quadraticChar_one_iff_isSquare ha

/-! ## Pointwise and existential squareness tests -/

/-- The Boolean squareness test at the representative `k`. Cost: one squaring and one equality
test in `ZMod p`. -/
def squareTest (a : ZMod p) (k : ℕ) : Bool := decide (((k : ℕ) : ZMod p) ^ 2 = a)

theorem squareTest_eq_true (a : ZMod p) (k : ℕ) :
    squareTest a k = true ↔ ((k : ℕ) : ZMod p) ^ 2 = a := by
  simp only [squareTest, decide_eq_true_eq]

/-- The Boolean squareness test over the representatives `0, …, n`. Recurrence
`g a 0 = squareTest a 0`, `g a (n + 1) = squareTest a (n + 1) || g a n`: at most `n + 1`
squarings and `n + 1` disjunctions in `Bool`, `O(1)` extra `Bool`s. -/
def anySquareTest (a : ZMod p) : ℕ → Bool
  | 0 => squareTest a 0
  | n + 1 => squareTest a (n + 1) || anySquareTest a n

theorem anySquareTest_eq_true (a : ZMod p) (n : ℕ) :
    anySquareTest a n = true ↔ ∃ k ≤ n, ((k : ℕ) : ZMod p) ^ 2 = a := by
  induction n with
  | zero =>
      simp only [anySquareTest, squareTest_eq_true]
      constructor
      · rintro h; exact ⟨0, Nat.zero_le _, h⟩
      · rintro ⟨k, hk, h⟩
        have hk0 : k = 0 := Nat.eq_zero_of_le_zero hk
        rw [hk0] at h; exact h
  | succ n ih =>
      simp only [anySquareTest, Bool.or_eq_true, squareTest_eq_true, ih]
      constructor
      · rintro (h | ⟨k, hk, h⟩)
        · exact ⟨n + 1, Nat.le_refl _, h⟩
        · exact ⟨k, Nat.le_succ_of_le hk, h⟩
      · rintro ⟨k, hk, h⟩
        by_cases hk' : k = n + 1
        · subst hk'
          exact Or.inl h
        · exact Or.inr ⟨k, Nat.le_of_lt_succ (by grind), h⟩

theorem isSquare_iff_exists_natCast (a : ZMod p) :
    IsSquare a ↔ ∃ k ≤ p - 1, ((k : ℕ) : ZMod p) ^ 2 = a := by
  constructor
  · rintro ⟨x, hx⟩
    refine ⟨x.val, Nat.le_sub_of_add_le (ZMod.val_lt x), ?_⟩
    rw [ZMod.natCast_zmod_val]
    simpa only [pow_two] using hx.symm
  · rintro ⟨k, _, h⟩
    exact (isSquare_iff_exists_sq a).mpr ⟨((k : ℕ) : ZMod p), h.symm⟩

/-- Total decision for squareness in the prime field, by exhaustion of the representatives
`0, …, p - 1`. Cost: at most `p` squarings and `p` disjunctions in `Bool`, counted in
multiplications and comparisons; `O(1)` extra `Bool`s. -/
def is_square_dec (a : ZMod p) : Decidable (IsSquare a) :=
  decidable_of_iff (anySquareTest a (p - 1))
    ((anySquareTest_eq_true a (p - 1)).trans (isSquare_iff_exists_natCast a).symm)

/-- Euler's criterion in this file's terms: a nonzero element is a square exactly when
`powZMod a (p / 2) = 1`. -/
theorem is_square_iff_euler (a : ZMod p) (ha : a ≠ 0) : IsSquare a ↔ powZMod a (p / 2) = 1 := by
  rw [powZMod_eq]
  exact ZMod.euler_criterion p ha

theorem is_square_iff_legendre (a : ZMod p) (ha : a ≠ 0) : IsSquare a ↔ legendre a = 1 :=
  (legendre_eq_one_iff a ha).symm

/-! ## Square roots -/

/-- The greatest representative `k ≤ n` with `k ^ 2 = a`, as an element of `ZMod p`, or `none`.
Recurrence `f a 0 = if 0 ^ 2 = a then some 0 else none`,
`f a (n + 1) = if ((n + 1 : ℕ) : ZMod p) ^ 2 = a then some ((n + 1 : ℕ) : ZMod p) else f a n`:
at most `n + 1` squarings and equality tests in `ZMod p`; the recursion closes because its
measure is the bound, decreasing by one per step; `O(1)` extra `ZMod p` values. -/
def rootUpTo (a : ZMod p) : ℕ → Option (ZMod p)
  | 0 => if (0 : ZMod p) ^ 2 = a then some 0 else none
  | n + 1 =>
      if ((n + 1 : ℕ) : ZMod p) ^ 2 = a then some ((n + 1 : ℕ) : ZMod p) else rootUpTo a n

theorem rootUpTo_sound {a : ZMod p} {n : ℕ} {r : ZMod p} (h : rootUpTo a n = some r) :
    r ^ 2 = a ∧ r.val ≤ n := by
  induction n generalizing r with
  | zero =>
      simp only [rootUpTo] at h
      split at h
      · rename_i heq
        have hr : r = 0 := Option.some.inj h.symm
        subst hr
        exact ⟨heq, by rw [ZMod.val_zero]⟩
      · exfalso
        exact absurd h.symm (by simp)
  | succ n ih =>
      simp only [rootUpTo] at h
      split at h
      · rename_i heq
        have hr : r = ((n + 1 : ℕ) : ZMod p) := Option.some.inj h.symm
        subst hr
        refine ⟨heq, ?_⟩
        rw [ZMod.val_natCast]
        exact Nat.mod_le _ _
      · obtain ⟨h1, h2⟩ := ih h
        exact ⟨h1, Nat.le_succ_of_le h2⟩

theorem rootUpTo_complete {a : ZMod p} {n k : ℕ} (hk : k ≤ n)
    (hk2 : ((k : ℕ) : ZMod p) ^ 2 = a) : ∃ r, rootUpTo a n = some r := by
  induction n generalizing k with
  | zero =>
      have hk0 : k = 0 := Nat.eq_zero_of_le_zero hk
      rw [hk0] at hk2
      refine ⟨0, ?_⟩
      rw [rootUpTo, if_pos (by simpa using hk2)]
  | succ n ih =>
      by_cases hkn : k < n
      · obtain ⟨r, hr⟩ := ih (k := k) (Nat.le_of_lt hkn) hk2
        rw [rootUpTo]
        split
        · exact ⟨_, rfl⟩
        · exact ⟨r, hr⟩
      · by_cases hk1 : k = n + 1
        · refine ⟨(n + 1 : ℕ), ?_⟩
          rw [rootUpTo, if_pos (hk1 ▸ hk2)]
        · obtain ⟨r, hr⟩ := ih (k := k) (by omega) hk2
          rw [rootUpTo]
          split
          · exact ⟨_, rfl⟩
          · exact ⟨r, hr⟩

/-- Total square root over the prime field: the greatest representative `k < p` with
`k ^ 2 = a`, or `none` when `a` is not a square. Cost: `p` squarings and `p` comparisons in
`ZMod p`; `O(1)` extra `ZMod p` values. This is the total decision `IsSquare` does not supply,
at exhaustive-search cost, not at Tonelli–Shanks cost. -/
def sqrt_mod (a : ZMod p) : Option (ZMod p) := rootUpTo a (p - 1)

theorem sqrt_mod_sound {a : ZMod p} {r : ZMod p} (h : sqrt_mod a = some r) : r ^ 2 = a :=
  (rootUpTo_sound h).1

theorem sqrt_mod_complete {a : ZMod p} (ha : IsSquare a) : ∃ r, sqrt_mod a = some r := by
  obtain ⟨k, hk, hk2⟩ := isSquare_iff_exists_natCast a |>.mp ha
  exact rootUpTo_complete hk hk2

/-- `sqrt_mod` returns the **greatest** representative `r` with `r ^ 2 = a`, so the converse of
`sqrt_mod_sound` is false as stated: at `p = 7`, `sqrt_mod 2 = some 4` while `3 ^ 2 = 2`.
Squaring over a prime field identifies a root only up to sign (`r ^ 2 = r' ^ 2` gives
`r = r'` or `r = -r'`), and `rootUpTo` selects the larger representative. The one-sided soundness
statement `sqrt_mod_sound` and the existence statement `sqrt_mod_complete` are what hold; a
biconditional needs the maximality side condition `∀ k, k ≤ p - 1 → k ^ 2 = a → r.val ≥ k.val`,
which this file does not state or prove. -/
theorem sqrt_mod_eq_some_sound {a r : ZMod p} (h : sqrt_mod a = some r) : r ^ 2 = a :=
  sqrt_mod_sound h

theorem sqrt_mod_eq_none_iff {a : ZMod p} : sqrt_mod a = none ↔ ¬ IsSquare a := by
  constructor
  · rintro h ha
    obtain ⟨r, hr⟩ := sqrt_mod_complete ha
    rw [h] at hr
    exact absurd hr.symm (by simp)
  · intro h
    by_contra hc
    obtain ⟨r, hr⟩ := Option.ne_none_iff_exists'.mp hc
    exact absurd ((isSquare_iff_exists_sq a).mpr ⟨r, (sqrt_mod_sound hr).symm⟩) h

/-! ## Tonelli–Shanks interface -/

/-- Tonelli–Shanks preconditions for `a` at `p`: the modulus shape `p - 1 = 2 ^ s * q` with
`q` odd, a quadratic nonresidue `z`, and a nonzero residue `a`. Euler's criterion is stated
with `p / 2`, which is `(p - 1) / 2` at an odd prime. -/
structure TonelliShanksInput (a : ZMod p) where
  /-- Exponent of the power of two in `p - 1`. -/
  s : ℕ
  /-- Odd part of `p - 1`. -/
  q : ℕ
  /-- The shape `p - 1 = 2 ^ s * q` the Tonelli–Shanks loop consumes. -/
  shape : p - 1 = 2 ^ s * q
  /-- The odd part is odd. -/
  q_odd : Odd q
  /-- A quadratic nonresidue at `p`. -/
  z : ZMod p
  /-- Euler's criterion on `z`, marking `z` a nonresidue. -/
  z_pow : z ^ (p / 2) = -1
  /-- A square root is requested for a nonzero element. -/
  a_ne_zero : a ≠ 0
  /-- Euler's criterion on `a`, marking `a` a residue. -/
  a_pow : a ^ (p / 2) = 1

/-- The Tonelli–Shanks square-root value. This module implements no Tonelli–Shanks inner loop
(no repeated halving of `m`, no search over candidate nonresidues), so the cost is the `O(p)`
search cost of `sqrt_mod` and not the `O(log ^ 2 p)` multiplication count of the classical
algorithm. -/
def sqrt_mod_TS (a : ZMod p) (_t : TonelliShanksInput a) : Option (ZMod p) := sqrt_mod a

/-- The Tonelli–Shanks conclusion: under the preconditions the returned value is a square
root of `a`, and it is returned rather than refused. The proof uses `a_ne_zero` and `a_pow`
only; `shape`, `q_odd` and `z_pow` are carried for an inner loop and this file proves nothing
about them. -/
theorem sqrt_mod_TS_correct {a : ZMod p} (t : TonelliShanksInput a) :
    ∃ r : ZMod p, r ^ 2 = a ∧ sqrt_mod_TS a t = some r := by
  obtain ⟨r, hr⟩ := sqrt_mod_complete ((is_square_iff_euler a t.a_ne_zero).mpr
    (by rw [powZMod_eq]; exact t.a_pow))
  exact ⟨r, sqrt_mod_sound hr, hr⟩

/-- A quadratic nonresidue exists at every odd prime, which is what supplies the field
`z_pow` of `TonelliShanksInput`. -/
theorem exists_nonresidue (hp2 : p ≠ 2) : ∃ z : ZMod p, z ^ (p / 2) = -1 := by
  have hchar : ringChar (ZMod p) ≠ 2 := by rw [ringChar_zmod_n]; exact hp2
  obtain ⟨z, hz⟩ := quadraticChar_exists_neg_one hchar
  refine ⟨z, ?_⟩
  have hz0 : z ≠ 0 := by
    intro hzz
    have h' : quadraticChar (ZMod p) z = -1 := hz
    rw [hzz, quadraticChar_zero] at h'
    norm_num at h'
  have hns : ¬ IsSquare z :=
    (quadraticChar_neg_one_iff_not_isSquare).mp hz
  exact (ZMod.pow_div_two_eq_neg_one_or_one p hz0).resolve_left (fun h =>
    hns ((is_square_iff_euler z hz0).mpr (by rw [powZMod_eq]; exact h)))

/-! ## Executable witnesses -/

-- The statements above are kernel-checked; the examples below run the definitions.

section Witnesses

local instance instFact7 : Fact (7 : ℕ).Prime := ⟨by decide⟩

example : (2 : ZMod 7) * invZMod (2 : ZMod 7) = 1 := by native_decide

example : sqrt_mod (2 : ZMod 7) = some ((4 : ℕ) : ZMod 7) := by native_decide

example : sqrt_mod (3 : ZMod 7) = none := by native_decide

example : legendre (2 : ZMod 7) = 1 := by native_decide

example : legendre (3 : ZMod 7) = -1 := by native_decide

example : anySquareTest (4 : ZMod 7) 6 = true := by native_decide

end Witnesses

end Apeiron.Crypto.Field