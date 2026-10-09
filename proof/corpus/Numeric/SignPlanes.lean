/-
Exact one-bit sign planes.

A `b`-bit nonnegative code `q` is exactly `b` sign planes: its signed value `2q − (2^b − 1)` is
the signed binary expansion `Σ_{k<b} 2^k (2·bit_k(q) − 1)` with every digit in `{−1, +1}`
(`signPlane_decomp`). The map taking a bit vector `s : Fin b → Bool` to the signed value
`Σ_k 2^k (2·[s k] − 1)` is injective and its image is exactly the odd integers of absolute value at
most `2^b − 1` (`planeVal_injective`, `planeVal_range`); the underlying unsigned value
`planeNat s = Σ_k 2^k [s k]` is the usual binary encoding, a bijection onto `Fin (2^b)`
(`planeNat_eq_finFunctionFinEquiv`, `planeNat_surj`, `planeNat_injective`). The matrix–vector form
holds entrywise: for a code matrix `Q` with every entry `< 2^b`,
`(2Q − (2^b − 1)) *ᵥ x = Σ_{k<b} 2^k • (S_k *ᵥ x)` with `S_k i j = 2·bit_k(Q i j) − 1 ∈ {±1}`
(`matvec_decomp`). Concrete `Q4_K`-shaped (`b = 4`) and `Q8_0`-shaped (`b = 8`, the affine reading
`2q + 1` of the signed code `q = byte − 128`) blocks are checked by `decide` (`q4k_instance`,
`q8_0_instance`).

No `sorry`, `axiom`, `maxHeartbeats 0`, `native_decide` or `bv_decide` appears; every `decide`
acts on closed literals only.
-/
import Mathlib

open scoped BigOperators
open scoped Matrix

namespace apeiron_word

namespace SignPlanes

/-! ## The signed binary decomposition of a code -/

/-- `Σ_{k<b} 2^k = 2^b − 1` in `ℤ`. -/
theorem geom_sum_two (b : ℕ) :
    (∑ k ∈ Finset.range b, (2 : ℤ) ^ k) = (2 : ℤ) ^ b - 1 := by
  induction b with
  | zero => norm_num
  | succ b ih => rw [Finset.sum_range_succ, ih, pow_succ]; ring

/-- `Σ_{k<b} 2^k = 2^b − 1` in `ℕ`. -/
theorem geom_sum_two_nat (b : ℕ) :
    (∑ k ∈ Finset.range b, (2 : ℕ) ^ k) = 2 ^ b - 1 := by
  induction b with
  | zero => norm_num
  | succ b ih => rw [Finset.sum_range_succ, ih, pow_succ]; omega

/-- `2^{b+1} = 2^b · 2`, so the top bit of `q % 2^{b+1}` splits off. -/
theorem nat_mod_mul_pow (q b : ℕ) :
    q % 2 ^ (b + 1) = q % 2 ^ b + 2 ^ b * (q / 2 ^ b % 2) := by
  rw [show (2 : ℕ) ^ (b + 1) = 2 ^ b * 2 by rw [pow_succ]]
  exact Nat.mod_mul

/-- The `k`-th sign `2·bit_k(q) − 1` is `2·(q / 2^k % 2) − 1`. -/
theorem testBit_term (q k : ℕ) :
    2 * (if q.testBit k then (1 : ℤ) else 0) - 1
      = 2 * ((q / 2 ^ k % 2 : ℕ) : ℤ) - 1 := by
  have hb : (if q.testBit k then (1 : ℤ) else 0) = ((q.testBit k).toNat : ℤ) := by
    cases q.testBit k <;> rfl
  rw [hb, Nat.toNat_testBit]

/-- The signed partial value of the low bits is `2·(q % 2^b) − (2^b − 1)`. -/
theorem sum_shifted_mod (q b : ℕ) :
    (∑ k ∈ Finset.range b, (2 : ℤ) ^ k * (2 * ((q / 2 ^ k % 2 : ℕ) : ℤ) - 1))
      = 2 * ((q % 2 ^ b : ℕ) : ℤ) - ((2 : ℤ) ^ b - 1) := by
  induction b with
  | zero => simp [Nat.mod_one]
  | succ b ih =>
      rw [Finset.sum_range_succ, ih]
      have hcast : (((q % 2 ^ (b + 1) : ℕ)) : ℤ)
          = ((q % 2 ^ b : ℕ) : ℤ) + (2 : ℤ) ^ b * ((q / 2 ^ b % 2 : ℕ) : ℤ) := by
        rw [nat_mod_mul_pow q b]; push_cast; ring
      rw [hcast, pow_succ]
      ring

/-- A `b`-bit code `q` is exactly `b` sign planes: `2q − (2^b − 1) = Σ_{k<b} 2^k (2·bit_k(q) − 1)`. -/
theorem signPlane_decomp {q b : ℕ} (hq : q < 2 ^ b) :
    (2 * (q : ℤ)) - ((2 : ℤ) ^ b - 1) =
      ∑ k ∈ Finset.range b, (2 : ℤ) ^ k * (2 * (if q.testBit k then 1 else 0) - 1) := by
  have hsum : (∑ k ∈ Finset.range b, (2 : ℤ) ^ k * (2 * (if q.testBit k then 1 else 0) - 1))
      = ∑ k ∈ Finset.range b, (2 : ℤ) ^ k * (2 * ((q / 2 ^ k % 2 : ℕ) : ℤ) - 1) :=
    Finset.sum_congr rfl fun k _ => by rw [testBit_term]
  rw [hsum, sum_shifted_mod, Nat.mod_eq_of_lt hq]

/-! ## The signed value map `(Fin b → Bool) → ℤ` -/

/-- The unsigned binary value of a bit vector, `Σ_k 2^k [s k]`. -/
def planeNat {b : ℕ} (s : Fin b → Bool) : ℕ :=
  ∑ i : Fin b, (if s i then 1 else 0) * 2 ^ (i : ℕ)

/-- The signed value of a bit vector, `Σ_k 2^k (2·[s k] − 1)`. -/
def planeVal {b : ℕ} (s : Fin b → Bool) : ℤ :=
  ∑ i : Fin b, (2 : ℤ) ^ (i : ℕ) * (2 * (if s i then 1 else 0) - 1)

/-- The bit vector of a `b`-bit code `q`. -/
def bitVec (b q : ℕ) : Fin b → Bool := fun i => q.testBit (i : ℕ)

/-- `b` bits as elements of `Fin 2`. -/
def bits {b : ℕ} (s : Fin b → Bool) : Fin b → Fin 2 := fun i => if s i then 1 else 0

/-- `planeNat` is the binary encoding `finFunctionFinEquiv` of `bits`. -/
theorem planeNat_eq_finFunctionFinEquiv (b : ℕ) (s : Fin b → Bool) :
    planeNat s = (finFunctionFinEquiv (bits s) : ℕ) := by
  rw [finFunctionFinEquiv_apply]
  unfold planeNat bits
  apply Finset.sum_congr rfl
  intro i _
  congr 1
  by_cases h : s i <;> simp [h]

/-- A single `Fin 2` is recovered from its comparison with `1`. -/
theorem bits_eq_of (g : Fin 2) : (if (g == 1) then (1 : Fin 2) else 0) = g := by
  fin_cases g <;> rfl

/-- `bits` is a bijection between `(Fin b → Bool)` and `(Fin b → Fin 2)`. -/
theorem bits_surjective (b : ℕ) : Function.Surjective (@bits b) :=
  fun g => ⟨fun i => (g i == 1), funext fun i => bits_eq_of (g i)⟩

/-- `planeNat` is injective. -/
theorem planeNat_injective (b : ℕ) : Function.Injective (@planeNat b) := by
  intro s t h
  have hs : (finFunctionFinEquiv (bits s)) = (finFunctionFinEquiv (bits t)) := by
    apply Fin.ext
    rw [← planeNat_eq_finFunctionFinEquiv, ← planeNat_eq_finFunctionFinEquiv, h]
  have hb : bits s = bits t := finFunctionFinEquiv.injective hs
  funext i
  have := congrFun hb i
  unfold bits at this
  by_cases hs' : s i <;> by_cases ht' : t i <;> simp_all

/-- Every `n < 2^b` is the unsigned value of a bit vector. -/
theorem planeNat_surj (b : ℕ) {n : ℕ} (hn : n < 2 ^ b) :
    ∃ s : Fin b → Bool, planeNat s = n := by
  obtain ⟨g, hg⟩ := finFunctionFinEquiv.surjective (⟨n, hn⟩ : Fin (2 ^ b))
  obtain ⟨s, hs⟩ := bits_surjective b g
  exact ⟨s, by rw [planeNat_eq_finFunctionFinEquiv, hs, hg]⟩

/-- `planeVal s = 2·planeNat s − (2^b − 1)`. -/
theorem planeVal_eq (b : ℕ) (s : Fin b → Bool) :
    planeVal s = 2 * (planeNat s : ℤ) - ((2 : ℤ) ^ b - 1) := by
  have hlin : planeVal s
      = 2 * (∑ i : Fin b, (2 : ℤ) ^ (i : ℕ) * ((if s i then 1 else 0 : ℤ)))
          - (∑ i : Fin b, (2 : ℤ) ^ (i : ℕ)) := by
    unfold planeVal
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hlin]
  have h2 : (∑ i : Fin b, (2 : ℤ) ^ (i : ℕ) * ((if s i then 1 else 0 : ℤ)))
      = (planeNat s : ℤ) := by
    unfold planeNat
    push_cast
    apply Finset.sum_congr rfl
    intro i _
    ring
  have h3 : (∑ i : Fin b, (2 : ℤ) ^ (i : ℕ)) = (2 : ℤ) ^ b - 1 := by
    rw [Fin.sum_univ_eq_sum_range (fun i => (2 : ℤ) ^ i) b, geom_sum_two]
  rw [h2, h3]

/-- `planeVal` is injective. -/
theorem planeVal_injective (b : ℕ) : Function.Injective (@planeVal b) := by
  intro s t h
  rw [planeVal_eq, planeVal_eq] at h
  have h2 : (planeNat s : ℤ) = (planeNat t : ℤ) := by linarith
  exact planeNat_injective b (Int.ofNat.inj h2)

/-- `planeNat s ≤ 2^b − 1`. -/
theorem planeNat_le (b : ℕ) (s : Fin b → Bool) : planeNat s ≤ 2 ^ b - 1 := by
  calc planeNat s ≤ ∑ i : Fin b, 2 ^ (i : ℕ) := by
        unfold planeNat
        apply Finset.sum_le_sum
        intro i _
        by_cases h : s i <;> simp [h]
    _ = 2 ^ b - 1 := by rw [Fin.sum_univ_eq_sum_range (fun i => (2 : ℕ) ^ i) b, geom_sum_two_nat]

/-- `planeNat s < 2^b`. -/
theorem planeNat_lt (b : ℕ) (s : Fin b → Bool) : planeNat s < 2 ^ b := by
  have h := planeNat_le b s
  have hp : 0 < 2 ^ b := pow_pos (by norm_num) b
  omega

/-- `2^b − 1` is odd for `b > 0`. -/
theorem odd_two_pow_sub_one (b : ℕ) (hb : 0 < b) : Odd ((2 : ℤ) ^ b - 1) := by
  obtain ⟨b', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hb)
  rw [pow_succ]
  exact ⟨(2 : ℤ) ^ b' - 1, by ring⟩

/-- `planeVal s` is odd for `b > 0`. -/
theorem planeVal_odd (b : ℕ) (hb : 0 < b) (s : Fin b → Bool) : Odd (planeVal s) := by
  obtain ⟨m, hm⟩ := odd_two_pow_sub_one b hb
  rw [planeVal_eq]
  refine ⟨(planeNat s : ℤ) - m - 1, ?_⟩
  rw [hm]; ring

/-- `|planeVal s| ≤ 2^b − 1`. -/
theorem planeVal_abs_le (b : ℕ) (s : Fin b → Bool) :
    |planeVal s| ≤ (2 : ℤ) ^ b - 1 := by
  have hlt : planeNat s < 2 ^ b := planeNat_lt b s
  have hle : (planeNat s : ℤ) ≤ (2 : ℤ) ^ b - 1 := by
    have hcast : ((2 ^ b - 1 : ℕ) : ℤ) = (2 : ℤ) ^ b - 1 := by
      rw [Nat.cast_sub (Nat.one_le_two_pow (n := b))]; push_cast; ring
    rw [← hcast]
    exact_mod_cast (by omega : planeNat s ≤ 2 ^ b - 1)
  have h0 : (0 : ℤ) ≤ (planeNat s : ℤ) := Int.natCast_nonneg _
  rw [planeVal_eq, abs_le]
  constructor <;> linarith

/-- Every odd integer of absolute value at most `2^b − 1` is a signed value. -/
theorem planeVal_surj (b : ℕ) (hb : 0 < b) {z : ℤ}
    (hodd : Odd z) (habs : |z| ≤ (2 : ℤ) ^ b - 1) :
    ∃ s : Fin b → Bool, planeVal s = z := by
  set M : ℤ := (2 : ℤ) ^ b - 1 with hM
  have hModd : Odd M := by rw [hM]; exact odd_two_pow_sub_one b hb
  have hzlow : -M ≤ z := (abs_le.mp habs).1
  have hzhigh : z ≤ M := (abs_le.mp habs).2
  have heven : Even (z + M) := by
    obtain ⟨a, ha⟩ := hodd
    obtain ⟨c, hc⟩ := hModd
    exact ⟨a + c + 1, by rw [ha, hc]; ring⟩
  have hEq : 2 * ((z + M) / 2) = z + M := Int.two_mul_ediv_two_of_even heven
  have hn0 : 0 ≤ (z + M) / 2 := Int.ediv_nonneg (by linarith) (by norm_num)
  have hnlt : (z + M) / 2 < (2 : ℤ) ^ b := by
    rw [Int.ediv_lt_iff_lt_mul (by norm_num : (0 : ℤ) < 2)]
    have : z + M ≤ M + M := by linarith
    rw [hM] at this ⊢
    linarith
  let n : ℕ := ((z + M) / 2).toNat
  have hnval : (n : ℤ) = (z + M) / 2 := Int.toNat_of_nonneg hn0
  have hnlt' : n < 2 ^ b := by
    have : ((n : ℤ)) < (2 : ℤ) ^ b := by rw [hnval]; exact hnlt
    exact_mod_cast this
  obtain ⟨s, hs⟩ := planeNat_surj b hnlt'
  refine ⟨s, ?_⟩
  rw [planeVal_eq, hs]
  have h2n : 2 * (n : ℤ) = z + M := by rw [hnval]; exact hEq
  linarith

/-- `planeVal` is a bijection onto the odd integers of absolute value at most `2^b − 1`. -/
theorem planeVal_range (b : ℕ) (hb : 0 < b) :
    Set.range (@planeVal b) = {z : ℤ | Odd z ∧ |z| ≤ (2 : ℤ) ^ b - 1} := by
  apply Set.ext
  intro z
  constructor
  · rintro ⟨s, rfl⟩
    exact ⟨planeVal_odd b hb s, planeVal_abs_le b s⟩
  · rintro ⟨hodd, habs⟩
    exact planeVal_surj b hb hodd habs

/-! ## Matrix–vector form -/

/-- The `k`-th sign plane of a code matrix: `S_k i j = 2·bit_k(Q i j) − 1`. -/
def signPlane {m n : ℕ} (Q : Matrix (Fin m) (Fin n) ℕ) (k : ℕ) : Matrix (Fin m) (Fin n) ℤ :=
  fun i j => 2 * (if (Q i j).testBit k then 1 else 0) - 1

/-- Every entry of a sign plane is `+1` or `−1`. -/
theorem signPlane_entries {m n : ℕ} (Q : Matrix (Fin m) (Fin n) ℕ) (k : ℕ) (i : Fin m)
    (j : Fin n) : signPlane Q k i j = 1 ∨ signPlane Q k i j = -1 := by
  unfold signPlane
  cases (Q i j).testBit k <;> simp

/-- The sign-plane matvec form: for a code matrix `Q` with every entry `< 2^b`,
`(2Q − (2^b − 1)) *ᵥ x = Σ_{k<b} 2^k • (S_k *ᵥ x)` with `S_k i j = 2·bit_k(Q i j) − 1 ∈ {±1}`. -/
theorem matvec_decomp {m n b : ℕ} (Q : Matrix (Fin m) (Fin n) ℕ)
    (hQ : ∀ i j, Q i j < 2 ^ b) (x : Fin n → ℤ) :
    (fun i j => 2 * (Q i j : ℤ) - ((2 : ℤ) ^ b - 1)) *ᵥ x
      = ∑ k ∈ Finset.range b, (2 : ℤ) ^ k • (signPlane Q k *ᵥ x) := by
  have hsum : (∑ k ∈ Finset.range b, (2 : ℤ) ^ k • signPlane Q k)
      = (fun i j => 2 * (Q i j : ℤ) - ((2 : ℤ) ^ b - 1)) := by
    funext i j
    rw [Matrix.sum_apply]
    simp only [Matrix.smul_apply, smul_eq_mul, signPlane]
    exact (signPlane_decomp (hQ i j)).symm
  rw [← hsum, Matrix.sum_mulVec]
  apply Finset.sum_congr rfl
  intro k _
  rw [Matrix.smul_mulVec]

/-! ## Concrete blocks -/

/-- A `Q4_K`-shaped block of eight 4-bit codes. -/
def q4kBlock : Fin 8 → ℕ := ![3, 7, 11, 15, 0, 5, 9, 14]

/-- A `Q8_0`-shaped block of eight stored bytes. -/
def q8_0Block : Fin 8 → ℕ := ![0, 1, 127, 128, 200, 255, 42, 100]

/-- Each of the eight 4-bit codes decomposes into four sign planes. -/
theorem q4k_instance :
    ∀ i : Fin 8, 2 * (q4kBlock i : ℤ) - ((2 : ℤ) ^ 4 - 1) =
      ∑ k ∈ Finset.range 4, (2 : ℤ) ^ k * (2 * (if (q4kBlock i).testBit k then 1 else 0) - 1) := by
  decide

/-- The `Q8_0` affine reading: the eight stored bytes give the plane value `2q + 1` of the signed
code `q = byte − 128`. -/
theorem q8_0_instance :
    ∀ i : Fin 8, planeVal (bitVec 8 (q8_0Block i)) = 2 * ((q8_0Block i : ℤ) - 128) + 1 := by
  decide

end SignPlanes

end apeiron_word
