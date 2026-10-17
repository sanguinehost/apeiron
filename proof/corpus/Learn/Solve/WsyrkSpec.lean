import ApeironWord
import Learn.Grad.DenseLinearSpec
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted `src/learn/solve/syrk.rs`: independent spec

The independent `ℤ`/`List ℤ` denotation the extracted weighted SYRK fold and its elementwise
Hadamard factor are refined against: the Q32.32 rounding by 2⁶⁴ (`rnd64`), the weighted exact
`i128` Gram accumulation `gsum`, the unit-weight function `wref` (`w = None` is raw `2³²`), the
row-major output model `gramOut`, and the checked-arithmetic lemmas the refinement consumes. No
extracted definition is referenced by the models here.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the vector container layout are not covered;
only the extracted integer body.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## 1. The Q32.32 rounding by 2⁶⁴ -/

/-- `(v + 2⁶³) >>> 64` — round-to-nearest of a Q96.96 value to Q32.32. -/
def rnd64 (v : ℤ) : ℤ := (v + 2 ^ 63) >>> 64

theorem rnd64_eq_div (v : ℤ) : rnd64 v = (v + 2 ^ 63) / 2 ^ 64 := by
  rw [rnd64, Int.shiftRight_eq_div_pow]
  norm_num

/-! ## 2. The weighted Gram accumulation -/

/-- `k` terms of `Σ_{t<k} W t · L[t·n + i] · L[t·n + j]`. -/
def gsum (L : List ℤ) (W : ℕ → ℤ) (n i j : ℕ) : ℕ → ℤ
  | 0 => 0
  | k + 1 => gsum L W n i j k + W k * L.getD (k * n + i) 0 * L.getD (k * n + j) 0

theorem gsum_succ (L : List ℤ) (W : ℕ → ℤ) (n i j k : ℕ) :
    gsum L W n i j (k + 1) =
      gsum L W n i j k + W k * L.getD (k * n + i) 0 * L.getD (k * n + j) 0 := rfl

/-- The `o`-term weighted Gram sum is symmetric in its two output indices. -/
theorem gsum_comm (L : List ℤ) (W : ℕ → ℤ) (n i j : ℕ) :
    ∀ k, gsum L W n i j k = gsum L W n j i k
  | 0 => rfl
  | k + 1 => by
    rw [gsum_succ, gsum_succ, gsum_comm L W n i j k]
    ring

/-- The weight function: `w = None` is the unit weight in raw Q32.32. -/
def wref (w : Option (Slice I64)) : ℕ → ℤ :=
  match w with
  | none => fun _ => 2 ^ 32
  | some ws => fun k => (qvals ws).getD k 0

theorem wref_none (k : ℕ) : wref none k = 2 ^ 32 := rfl

theorem wref_some (ws : Slice I64) (k : ℕ) : wref (some ws) k = (qvals ws).getD k 0 := rfl

/-- The reference Gram entry: `round64(Σ_{r<o} w_r · x[r,i] · x[r,j])` in Q32.32. -/
def gRef (x : Slice I64) (w : Option (Slice I64)) (o n i j : ℕ) : ℤ :=
  rnd64 (gsum (qvals x) (wref w) n i j o)

/-- The reference Gram is symmetric: the Rust mirror writes one value to both entries. -/
theorem gRef_symm (x : Slice I64) (w : Option (Slice I64)) (o n i j : ℕ) :
    gRef x w o n i j = gRef x w o n j i := by
  unfold gRef
  rw [gsum_comm]

/-- The row-major `n × n` output model. -/
def gramOut (x : Slice I64) (w : Option (Slice I64)) (o n : ℕ) : List ℤ :=
  rowsOf n (gRef x w o n) n

theorem gramOut_length (x : Slice I64) (w : Option (Slice I64)) (o n : ℕ) :
    (gramOut x w o n).length = n * n := by
  rw [gramOut, rowsOf_length]

theorem gramOut_getD (x : Slice I64) (w : Option (Slice I64)) (o n : ℕ) {p c : ℕ}
    (hp : p < n) (hc : c < n) :
    (gramOut x w o n).getD (p * n + c) 0 = gRef x w o n p c :=
  rowsOf_getD n (gRef x w o n) n p c hp hc

/-! ## 3. The magnitude bounds -/

/-- A single value strictly inside `±B`. -/
def absBound (v : ℤ) (B : ℕ) : Prop := -(B : ℤ) < v ∧ v < (B : ℤ)

/-- Every entry of a `ℤ` list (read through `getD`) is strictly inside `±2³⁶`. -/
def listBound (L : List ℤ) : Prop := ∀ p, absBound (L.getD p 0) (2 ^ 36)

/-- A weight function bounded by `2³⁶` (the unit weight `2³²` and every accepted `w` entry). -/
def wfunBound (W : ℕ → ℤ) : Prop := ∀ k, -(2 : ℤ) ^ 36 ≤ W k ∧ W k ≤ (2 : ℤ) ^ 36

theorem listBound_le (L : List ℤ) (h : listBound L) (p : ℕ) :
    |L.getD p 0| ≤ (2 : ℤ) ^ 36 := by
  obtain ⟨h1, h2⟩ := h p
  rw [abs_le]
  omega

theorem wfunBound_abs (W : ℕ → ℤ) (h : wfunBound W) (k : ℕ) :
    |W k| ≤ (2 : ℤ) ^ 36 := by
  obtain ⟨h1, h2⟩ := h k
  rw [abs_le]
  exact ⟨h1, h2⟩

/-- Every term of the weighted sum is bounded by `2³⁶ · 2³⁶ · 2³⁶ = 2¹⁰⁸`. -/
theorem gsum_abs_le (L : List ℤ) (W : ℕ → ℤ) (n i j : ℕ)
    (hL : listBound L) (hW : wfunBound W) :
    ∀ k, |gsum L W n i j k| ≤ (k : ℤ) * 2 ^ 108 := by
  intro k
  induction k with
  | zero => simp [gsum]
  | succ t ih =>
    rw [gsum_succ]
    have hL1 : |L.getD (t * n + i) 0| ≤ (2 : ℤ) ^ 36 := listBound_le L hL _
    have hL2 : |L.getD (t * n + j) 0| ≤ (2 : ℤ) ^ 36 := listBound_le L hL _
    have hw := wfunBound_abs W hW t
    have h12 : |L.getD (t * n + i) 0| * |L.getD (t * n + j) 0| ≤ (2 : ℤ) ^ 72 := by
      have h := mul_le_mul hL1 hL2 (abs_nonneg _) (by norm_num : (0 : ℤ) ≤ 2 ^ 36)
      norm_num at h ⊢
      exact h
    have hterm : |W t * L.getD (t * n + i) 0 * L.getD (t * n + j) 0| ≤ (2 : ℤ) ^ 108 := by
      have h := mul_le_mul hw h12 (mul_nonneg (abs_nonneg _) (abs_nonneg _))
        (by norm_num : (0 : ℤ) ≤ 2 ^ 36)
      rw [abs_mul, abs_mul, mul_assoc]
      norm_num at h ⊢
      exact h
    calc |gsum L W n i j t + W t * L.getD (t * n + i) 0 * L.getD (t * n + j) 0|
        ≤ |gsum L W n i j t| + |W t * L.getD (t * n + i) 0 * L.getD (t * n + j) 0| :=
          abs_add_le _ _
      _ ≤ (t : ℤ) * 2 ^ 108 + 2 ^ 108 := by linarith [ih, hterm]
      _ = ((t : ℤ) + 1) * 2 ^ 108 := by ring
      _ = ((t + 1 : ℕ) : ℤ) * 2 ^ 108 := by push_cast; ring

/-- Under `o < 2¹⁸` every prefix of the accumulation is inside `±2¹²⁶`, hence `i128`. -/
theorem gsum_lt_126 (L : List ℤ) (W : ℕ → ℤ) (n i j o k : ℕ)
    (hL : listBound L) (hW : wfunBound W) (ho : o < 2 ^ 18) (hk : k ≤ o) :
    |gsum L W n i j k| < (2 : ℤ) ^ 126 := by
  have h := gsum_abs_le L W n i j hL hW k
  have hk' : (k : ℤ) ≤ (o : ℤ) := by exact_mod_cast hk
  have ho' : (o : ℤ) < (2 : ℤ) ^ 18 := by exact_mod_cast ho
  have hk18 : (k : ℤ) < (2 : ℤ) ^ 18 := lt_of_le_of_lt hk' ho'
  linarith [h, hk18]

theorem gsum_i128 (L : List ℤ) (W : ℕ → ℤ) (n i j o k : ℕ)
    (hL : listBound L) (hW : wfunBound W) (ho : o < 2 ^ 18) (hk : k ≤ o) :
    I128.min ≤ gsum L W n i j k ∧ gsum L W n i j k ≤ I128.max := by
  have h := gsum_lt_126 L W n i j o k hL hW ho hk
  obtain ⟨h1, h2⟩ := abs_lt.mp h
  rw [I128.min_eq, I128.max_eq]
  exact ⟨by linarith, by linarith⟩

/-- The rounding add `v + 2⁶³` stays inside `i128` when `|v| < 2¹²⁶`. -/
theorem rnd64_add_i128 (v : ℤ) (hv : |v| < (2 : ℤ) ^ 126) :
    I128.min ≤ v + 2 ^ 63 ∧ v + 2 ^ 63 ≤ I128.max := by
  obtain ⟨h1, h2⟩ := abs_lt.mp hv
  rw [I128.min_eq, I128.max_eq]
  exact ⟨by linarith, by linarith⟩

/-- The rounded Q32.32 value narrows into `i64` when `|v| < 2¹²⁶`. -/
theorem rnd64_fits_i64 (v : ℤ) (hv : |v| < (2 : ℤ) ^ 126) :
    -((2 : ℤ) ^ 63) ≤ rnd64 v ∧ rnd64 v ≤ (2 : ℤ) ^ 63 - 1 := by
  obtain ⟨h1, h2⟩ := abs_lt.mp hv
  have hup : v + 2 ^ 63 ≤ (2 : ℤ) ^ 127 - 1 := by
    have h : (2 : ℤ) ^ 126 + 2 ^ 63 < (2 : ℤ) ^ 127 := by norm_num
    linarith
  have hlow : -((2 : ℤ) ^ 127) ≤ v + 2 ^ 63 := by linarith
  rw [rnd64_eq_div]
  have hp : (0 : ℤ) < 2 ^ 64 := by norm_num
  have hcu := Int.ediv_le_ediv (c := (2 : ℤ) ^ 64) hp hup
  have hcl := Int.ediv_le_ediv (c := (2 : ℤ) ^ 64) hp hlow
  have htop : ((2 : ℤ) ^ 127 - 1) / 2 ^ 64 = (2 : ℤ) ^ 63 - 1 := by norm_num
  have hbot : (-((2 : ℤ) ^ 127)) / 2 ^ 64 = -((2 : ℤ) ^ 63) := by norm_num
  rw [htop] at hcu
  rw [hbot] at hcl
  exact ⟨hcl, hcu⟩

/-- The three domain facts the SYRK refinement consumes, from the accepted-domain bounds. -/
theorem gsum_domain (L : List ℤ) (W : ℕ → ℤ) (n i j o : ℕ)
    (hL : listBound L) (hW : wfunBound W) (ho : o < 2 ^ 18) :
    (∀ k, k ≤ o → I128.min ≤ gsum L W n i j k ∧ gsum L W n i j k ≤ I128.max) ∧
    I128.min ≤ gsum L W n i j o + 2 ^ 63 ∧ gsum L W n i j o + 2 ^ 63 ≤ I128.max ∧
    -((2 : ℤ) ^ 63) ≤ rnd64 (gsum L W n i j o) ∧
      rnd64 (gsum L W n i j o) ≤ (2 : ℤ) ^ 63 - 1 := by
  have hlt := gsum_lt_126 L W n i j o o hL hW ho (le_refl o)
  obtain ⟨ha, hb⟩ := rnd64_add_i128 (gsum L W n i j o) hlt
  obtain ⟨hc, hd⟩ := rnd64_fits_i64 (gsum L W n i j o) hlt
  exact ⟨fun k hk => gsum_i128 L W n i j o k hL hW ho hk, ha, hb, hc, hd⟩

/-! ## 4. The elementwise Hadamard factor -/

/-- `k` entries of the in-place Hadamard fold: the first `k` entries rounded, the rest original. -/
def hadStep (L H : List ℤ) (k : ℕ) : List ℤ :=
  pushN [] (fun t => rnd32 (L.getD t 0 * H.getD t 0)) k ++ L.drop k

/-- One in-place write of the Hadamard fold advances the model, while `k` is in range. -/
theorem hadStep_succ (L H : List ℤ) (k : ℕ) (hk : k < L.length) :
    (hadStep L H k).set k (rnd32 (L.getD k 0 * H.getD k 0)) = hadStep L H (k + 1) := by
  have hne : L.drop k ≠ [] := by
    intro h
    rw [List.drop_eq_nil_iff] at h
    omega
  unfold hadStep
  rw [pushN]
  rw [List.set_append_right k (rnd32 (L.getD k 0 * H.getD k 0)) (by rw [pushN_length]; simp)]
  have h0 : k - (pushN ([] : List ℤ) (fun t => rnd32 (L.getD t 0 * H.getD t 0)) k).length = 0 := by
    rw [pushN_length]; simp
  rw [h0]
  cases hd : L.drop k with
  | nil => exact absurd hd hne
  | cons a t =>
    have ht : t = L.drop (k + 1) := by
      have h := List.tail_drop (l := L) (i := k)
      rw [hd] at h
      simpa using h
    rw [ht, List.set_cons_zero]
    simp [List.append_assoc]

theorem hadStep_length (L H : List ℤ) (k : ℕ) (hk : k ≤ L.length) :
    (hadStep L H k).length = L.length := by
  have hlen : (pushN ([] : List ℤ) (fun t => rnd32 (L.getD t 0 * H.getD t 0)) k).length = k := by
    rw [pushN_length]; simp
  unfold hadStep
  rw [List.length_append, hlen, List.length_drop]
  omega

theorem hadStep_getD_lt (L H : List ℤ) (k : ℕ) {j : ℕ} (hj : j < k) :
    (hadStep L H k).getD j 0 = rnd32 (L.getD j 0 * H.getD j 0) := by
  have hlen : (pushN ([] : List ℤ) (fun t => rnd32 (L.getD t 0 * H.getD t 0)) k).length = k := by
    rw [pushN_length]; simp
  unfold hadStep
  rw [List.getD_append _ _ _ _ (by rw [hlen]; exact hj)]
  simpa using pushN_getD_push [] (fun t => rnd32 (L.getD t 0 * H.getD t 0)) k j hj

/-- At the write position the fold still reads the original entry (in-range index `k`). -/
theorem hadStep_getD_self (L H : List ℤ) (k : ℕ) (hk : k < L.length) :
    (hadStep L H k).getD k 0 = L.getD k 0 := by
  have hlen : (pushN ([] : List ℤ) (fun t => rnd32 (L.getD t 0 * H.getD t 0)) k).length = k := by
    rw [pushN_length]; simp
  unfold hadStep
  rw [List.getD_append_right _ _ _ _ (by rw [hlen])]
  have h0 : k - (pushN ([] : List ℤ) (fun t => rnd32 (L.getD t 0 * H.getD t 0)) k).length = 0 := by
    rw [hlen, Nat.sub_self]
  rw [h0]
  have hne : L.drop k ≠ [] := by
    intro h
    rw [List.drop_eq_nil_iff] at h
    omega
  cases hd : L.drop k with
  | nil => exact absurd hd hne
  | cons a t =>
    have ha : a = L.getD k 0 := by
      have h := List.drop_eq_getElem_cons (l := L) hk
      rw [hd] at h
      have h1 : a = L[k] := (List.cons.inj h).1
      rw [h1, List.getD_eq_getElem L 0 hk]
    rw [ha]
    rfl

/-- The elementwise product-round-narrow at index `k` leaves every intermediate in range. -/
def hadFit (L H : List ℤ) (k : ℕ) : Prop :=
  I128.min ≤ L.getD k 0 * H.getD k 0 + 2 ^ 31 ∧
  L.getD k 0 * H.getD k 0 + 2 ^ 31 ≤ I128.max ∧
  -((2 : ℤ) ^ 63) ≤ rnd32 (L.getD k 0 * H.getD k 0) ∧
  rnd32 (L.getD k 0 * H.getD k 0) ≤ (2 : ℤ) ^ 63 - 1

end apeiron_word
