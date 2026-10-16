import ApeironWord
import Learn.Grad.DenseLinearSpec
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation

/-!
# The extracted `src/learn/solve/gemm.rs` Q32.32 matrix products: independent spec

The independent `ℤ`/`List ℤ` denotation the extracted blocked Q32.32 products `gemm_nn` and `gemm_nt`
are refined against: entry `(i, j)` of the row-major `m × n` output is the Q32.32 rounding
`rnd32` of the exact contraction `Σ_{l<k} A[i,l]·B[l,j]` (`nn`, `B` row-major `[k, n]`) or
`Σ_{l<k} A[i,l]·B[j,l]` (`nt`, `B` row-major `[n, k]`). `rnd32` and `dsum` are reused from
`Learn.Grad.DenseLinearSpec`; this module states WHAT each product computes, not how its loops run.

The magnitude contract: every entry of `a` and `b` is below `2³⁶` in raw units, and the contraction
length satisfies `k < MAXK = 2⁵⁵`; the exact `i128` contraction sum then lies inside `i128`
(`|Σ| ≤ k·(2³⁶−1)²`), the round-half add `2³¹` included. `withinBound` states the entry bound and
`gemmRefuse` the refusal chain of `solve_port.gemm.check`, in the order the source checks it.

Trust boundary: rustc/LLVM codegen, the kernel ABI, the `Vec`/`Slice` container layout and the
threaded driver are not covered; only the extracted integer body.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## 1. The entry bound and the refusal chain -/

/-- The Q32.32 entry magnitude bound `2³⁶` (`solve_port.syrk.BOUND_BITS = 36`). -/
def BND : ℤ := (2 : ℤ) ^ 36

/-- The largest accepted contraction length, `2⁵⁵ = 1 << (127 − 2·36)`. -/
def MAXK : ℤ := (2 : ℤ) ^ 55

/-- Every entry of a value list is below `2³⁶` in magnitude. -/
def withinBound (L : List ℤ) : Prop := ∀ z ∈ L, |z| < BND

/-- The reference refusal chain of `solve_port.gemm.check`, in the source's check order:
zero dimension, then `k` at the length bound, then the three dimension products, then the two slice
lengths, then the entry magnitudes. `none` is the accepted shape. -/
def gemmRefuse (la lb : ℕ) (m k n : ℕ) (bnd : Bool) : Option solve_port.gemm.GemmError :=
  if m = 0 ∨ k = 0 ∨ n = 0 then some .ZeroDim
  else if MAXK ≤ k then some .Overflow
  else if ¬ (m * k ≤ Usize.max ∧ k * n ≤ Usize.max ∧ n * m ≤ Usize.max) then some .Overflow
  else if ¬ (la = m * k ∧ lb = k * n) then some .ShapeMismatch
  else if ¬ bnd then some .InputBound
  else none

/-! ## 2. The pure `ℤ` references -/

/-- Entry `(i, j)` of `gemm_nn`: `rnd32(Σ_{l<k} A[i,l]·B[l,j])`, `B` row-major `[k, n]`. -/
def gnn (A B : List ℤ) (k n : ℕ) (i j : ℕ) : ℤ :=
  rnd32 (dsum A B (fun i _ l => i * k + l) (fun _ j l => l * n + j) i j k)

/-- Entry `(i, j)` of `gemm_nt`: `rnd32(Σ_{l<k} A[i,l]·B[j,l])`, `B` row-major `[n, k]`. -/
def gnt (A B : List ℤ) (k : ℕ) (i j : ℕ) : ℤ :=
  rnd32 (dsum A B (fun i _ l => i * k + l) (fun _ j l => j * k + l) i j k)

/-- The `gemm_nn` row-major `m × n` output. -/
def outNn (A B : List ℤ) (m k n : ℕ) : List ℤ := rowsOf n (gnn A B k n) m

/-- The `gemm_nt` row-major `m × n` output. -/
def outNt (A B : List ℤ) (m k n : ℕ) : List ℤ := rowsOf n (gnt A B k) m

theorem outNn_length (A B : List ℤ) (m k n : ℕ) : (outNn A B m k n).length = m * n :=
  rowsOf_length n (gnn A B k n) m

theorem outNt_length (A B : List ℤ) (m k n : ℕ) : (outNt A B m k n).length = m * n :=
  rowsOf_length n (gnt A B k) m

/-- Entry `(i, j)` of the `gemm_nn` output model. -/
theorem outNn_getD (A B : List ℤ) (m k n : ℕ) {i j : ℕ} (hi : i < m) (hj : j < n) :
    (outNn A B m k n).getD (i * n + j) 0 = gnn A B k n i j :=
  rowsOf_getD n (gnn A B k n) m i j hi hj

/-- Entry `(i, j)` of the `gemm_nt` output model. -/
theorem outNt_getD (A B : List ℤ) (m k n : ℕ) {i j : ℕ} (hi : i < m) (hj : j < n) :
    (outNt A B m k n).getD (i * n + j) 0 = gnt A B k i j :=
  rowsOf_getD n (gnt A B k) m i j hi hj

/-! ## 3. The contraction-bound arithmetic (no `i128` overflow) -/

/-- A past-the-end contraction read is `0`; an in-range one is a list entry. -/
theorem bnd_getD (L : List ℤ) (h : withinBound L) (i : ℕ) : |L.getD i 0| < BND := by
  rcases Nat.lt_or_ge i L.length with hi | hi
  · rw [List.getD_eq_getElem _ _ hi]
    exact h _ (List.getElem_mem hi)
  · rw [List.getD_eq_default _ _ hi]
    simp [BND]

/-- One product of two bounded entries is below `2⁷²`. -/
theorem bnd_mul (L R : List ℤ) (hL : withinBound L) (hR : withinBound R) (i j : ℕ) :
    |L.getD i 0 * R.getD j 0| < (2 : ℤ) ^ 72 := by
  rw [abs_mul]
  have hx := bnd_getD L hL i
  have hy := bnd_getD R hR j
  have hx' : |L.getD i 0| ≤ (2 : ℤ) ^ 36 - 1 := by rw [BND] at hx; omega
  have hy' : |R.getD j 0| ≤ (2 : ℤ) ^ 36 - 1 := by rw [BND] at hy; omega
  have hx0 : 0 ≤ |L.getD i 0| := abs_nonneg _
  have hy0 : 0 ≤ |R.getD j 0| := abs_nonneg _
  calc |L.getD i 0| * |R.getD j 0|
      ≤ ((2 : ℤ) ^ 36 - 1) * ((2 : ℤ) ^ 36 - 1) := by nlinarith
    _ < (2 : ℤ) ^ 72 := by norm_num

/-- The exact contraction of `k` bounded products is at most `k·(2³⁶−1)²`. -/
theorem dsum_abs_le (L R : List ℤ) (gL gR : ℕ → ℕ → ℕ → ℕ)
    (hL : withinBound L) (hR : withinBound R) (a b : ℕ) :
    ∀ k : ℕ, |dsum L R gL gR a b k| ≤ (k : ℤ) * ((2 : ℤ) ^ 36 - 1) ^ 2 := by
  intro k
  induction k with
  | zero => simp [dsum]
  | succ k ih =>
    rw [dsum_succ]
    have hterm : |L.getD (gL a b k) 0 * R.getD (gR a b k) 0| ≤ ((2 : ℤ) ^ 36 - 1) ^ 2 := by
      rw [abs_mul]
      have hx := bnd_getD L hL (gL a b k)
      have hy := bnd_getD R hR (gR a b k)
      have hx' : |L.getD (gL a b k) 0| ≤ (2 : ℤ) ^ 36 - 1 := by rw [BND] at hx; omega
      have hy' : |R.getD (gR a b k) 0| ≤ (2 : ℤ) ^ 36 - 1 := by rw [BND] at hy; omega
      have hx0 : 0 ≤ |L.getD (gL a b k) 0| := abs_nonneg _
      have hy0 : 0 ≤ |R.getD (gR a b k) 0| := abs_nonneg _
      nlinarith [sq_nonneg ((2 : ℤ) ^ 36 - 1)]
    calc |dsum L R gL gR a b k + L.getD (gL a b k) 0 * R.getD (gR a b k) 0|
        ≤ |dsum L R gL gR a b k| + |L.getD (gL a b k) 0 * R.getD (gR a b k) 0| :=
          abs_add_le _ _
      _ ≤ (k : ℤ) * ((2 : ℤ) ^ 36 - 1) ^ 2 + ((2 : ℤ) ^ 36 - 1) ^ 2 := by
          linarith
      _ = ((k + 1 : ℕ) : ℤ) * ((2 : ℤ) ^ 36 - 1) ^ 2 := by push_cast; ring

/-- Under `k < 2⁵⁵` the exact contraction is below `2¹²⁷`. -/
theorem dsum_abs_lt (L R : List ℤ) (gL gR : ℕ → ℕ → ℕ → ℕ)
    (hL : withinBound L) (hR : withinBound R) (a b k : ℕ) (hk : k < 2 ^ 55) :
    |dsum L R gL gR a b k| < (2 : ℤ) ^ 127 := by
  have h := dsum_abs_le L R gL gR hL hR a b k
  have hk' : (k : ℤ) ≤ (2 : ℤ) ^ 55 - 1 := by
    have : (k : ℤ) < (2 : ℤ) ^ 55 := by exact_mod_cast hk
    omega
  have hs : (0 : ℤ) ≤ ((2 : ℤ) ^ 36 - 1) ^ 2 := sq_nonneg _
  have hkey : ((2 : ℤ) ^ 55 - 1) * ((2 : ℤ) ^ 36 - 1) ^ 2 < (2 : ℤ) ^ 127 := by norm_num
  nlinarith [mul_le_mul_of_nonneg_right hk' hs]

/-- With `|s| ≤ (2⁵⁵−1)·(2³⁶−1)²`, the round-half add `2³¹` stays inside `i128`. -/
theorem round_half_fits (s : ℤ)
    (h : |s| ≤ ((2 : ℤ) ^ 55 - 1) * ((2 : ℤ) ^ 36 - 1) ^ 2) :
    I128.min ≤ s + 2 ^ 31 ∧ s + 2 ^ 31 ≤ I128.max := by
  rw [I128.min_eq, I128.max_eq]
  have hM : |s| + 2 ^ 31 < (2 : ℤ) ^ 127 := by
    have hc : ((2 : ℤ) ^ 55 - 1) * ((2 : ℤ) ^ 36 - 1) ^ 2 + 2 ^ 31 < (2 : ℤ) ^ 127 := by
      norm_num
    linarith
  have habs : |s + 2 ^ 31| ≤ |s| + 2 ^ 31 := by
    have h' := abs_add_le s ((2 : ℤ) ^ 31)
    rw [abs_of_nonneg (by norm_num : (0 : ℤ) ≤ (2 : ℤ) ^ 31)] at h'
    exact h'
  constructor
  · have hlt : -((2 : ℤ) ^ 127) < s + 2 ^ 31 := by
      have h1 : -|s + 2 ^ 31| ≤ s + 2 ^ 31 := neg_abs_le _
      linarith
    omega
  · have hlt : s + 2 ^ 31 < (2 : ℤ) ^ 127 := by linarith [le_abs_self s]
    omega

/-- The exact contraction sum plus the round-half add is inside `i128` under `k < 2⁵⁵`. -/
theorem dsum_round_fits (L R : List ℤ) (gL gR : ℕ → ℕ → ℕ → ℕ)
    (hL : withinBound L) (hR : withinBound R) (a b k : ℕ) (hk : k < 2 ^ 55) :
    I128.min ≤ dsum L R gL gR a b k + 2 ^ 31 ∧
      dsum L R gL gR a b k + 2 ^ 31 ≤ I128.max := by
  refine round_half_fits _ ?_
  have h := dsum_abs_le L R gL gR hL hR a b k
  have hk' : (k : ℤ) ≤ (2 : ℤ) ^ 55 - 1 := by
    have : (k : ℤ) < (2 : ℤ) ^ 55 := by exact_mod_cast hk
    omega
  calc |dsum L R gL gR a b k| ≤ (k : ℤ) * ((2 : ℤ) ^ 36 - 1) ^ 2 := h
    _ ≤ ((2 : ℤ) ^ 55 - 1) * ((2 : ℤ) ^ 36 - 1) ^ 2 :=
        mul_le_mul_of_nonneg_right hk' (sq_nonneg _)

/-- `MAXK` is `2⁵⁵`. -/
theorem MAXK_eq : MAXK = (2 : ℤ) ^ 55 := rfl

/-- `BND` is `2³⁶`. -/
theorem BND_eq : BND = (2 : ℤ) ^ 36 := rfl

end apeiron_word
