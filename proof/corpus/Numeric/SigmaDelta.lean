/-
First-order sigma-delta (error-feedback) rounding of a sequence `w` to `{-a, +a}`
(the ternary-hadamard note Part II).

State `e_0 = 0`, `e_{t+1} = e_t + w_t - a·s_t` with `s_t = sign(w_t + e_t)` (sign of the state-corrected
input, `+1` at `0`). Quantised value `q_t = a·s_t`, residual `r_t = w_t - q_t = e_{t+1} - e_t`.

Landed here, all over a linearly ordered field:

* `sd_state_bound`: `|w_t| ≤ a` for all `t` gives `|e_t| ≤ a`;
* `sd_prefix`: `Σ_{t<T} r_t = e_T` (the residual telescopes);
* `sd_abel`: exact summation by parts `Σ_{t<n} r_t x_t = e_n x_{n-1} - Σ_{1≤t<n} e_t (x_t - x_{t-1})`;
* `sd_dot_bound`: `|Σ r_t x_t| ≤ a·(|x_{n-1}| + Σ_{1≤t<n} |x_t - x_{t-1}|)`;
* `shape_dot`, `shape_null_exact`: `(U ε)·x = ε·(Uᵀ x)`, and zero when `ε` and `Uᵀ x` have disjoint support;
* `sd_beats_sign`: over `ℚ`, `w ≡ 1/3`, `a = 1`, `x ≡ 1`, `n = 6`: the sigma-delta residual dot
  is strictly smaller in absolute value than the plain-sign one.

Grade: `proved` for the statements above over an arbitrary ordered field (reals included); the
module is a pure algebra statement and does not read any Rust or extracted body. Trust boundary:
no claim about rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V, or the `unsafe` container; no
claim about model quality, only the worst-case dot-product error of the rounding scheme.
-/
import Mathlib

namespace Numeric.SigmaDelta

open Finset
open scoped Matrix

section Core

variable {α : Type*} [Field α] [LinearOrder α] [IsStrictOrderedRing α]

/-- Sign of the state-corrected input `w t + e`: `+1` at `0` and above, `-1` below. -/
def sdSignAt (w : ℕ → α) (t : ℕ) (e : α) : α := if 0 ≤ w t + e then 1 else -1

/-- Error-feedback state: `e_0 = 0`, `e_{t+1} = e_t + w_t - a·sign(w_t + e_t)`. -/
def sdState (w : ℕ → α) (a : α) : ℕ → α
  | 0 => 0
  | t + 1 => sdState w a t + w t - a * sdSignAt w t (sdState w a t)

/-- The sign emitted at step `t`; it reads the current state. -/
def sdSign (w : ℕ → α) (a : α) (t : ℕ) : α := sdSignAt w t (sdState w a t)

/-- Quantised value `a·s_t`. -/
def sdQ (w : ℕ → α) (a : α) (t : ℕ) : α := a * sdSign w a t

/-- Residual `w_t - q_t`. -/
def sdR (w : ℕ → α) (a : α) (t : ℕ) : α := w t - sdQ w a t

omit [IsStrictOrderedRing α] in
theorem sdState_succ (w : ℕ → α) (a : α) (t : ℕ) :
    sdState w a (t + 1) = sdState w a t + w t - a * sdSign w a t := rfl

omit [IsStrictOrderedRing α] in
theorem sdR_eq (w : ℕ → α) (a : α) (t : ℕ) :
    sdR w a t = sdState w a (t + 1) - sdState w a t := by
  rw [sdState_succ]; unfold sdR sdQ; ring

theorem sd_state_bound (w : ℕ → α) (a : α) (ha : 0 ≤ a) (hw : ∀ t, |w t| ≤ a) :
    ∀ t, |sdState w a t| ≤ a := by
  intro t
  induction t with
  | zero => simpa [sdState] using ha
  | succ t ih =>
    rw [sdState_succ]
    obtain ⟨h1, h2⟩ := abs_le.mp ih
    obtain ⟨h3, h4⟩ := abs_le.mp (hw t)
    unfold sdSign sdSignAt
    rw [abs_le]
    split_ifs with h
    · constructor <;> linarith
    · have h' := not_le.mp h
      constructor <;> linarith

omit [IsStrictOrderedRing α] in
theorem sd_prefix (w : ℕ → α) (a : α) (T : ℕ) :
    ∑ t ∈ range T, sdR w a t = sdState w a T := by
  induction T with
  | zero => simp [sdState]
  | succ T ih => rw [sum_range_succ, ih, sdR_eq]; ring

omit [IsStrictOrderedRing α] in
theorem sd_abel (w : ℕ → α) (a : α) (x : ℕ → α) (n : ℕ) (hn : 1 ≤ n) :
    ∑ t ∈ range n, sdR w a t * x t
      = sdState w a n * x (n - 1) - ∑ t ∈ Ico 1 n, sdState w a t * (x t - x (t - 1)) := by
  induction n, hn using Nat.le_induction with
  | base => simp [sdR_eq, sdState]
  | succ n hn ih =>
    rw [sum_range_succ, ih, sum_Ico_succ_top hn, sdR_eq]
    simp only [Nat.add_sub_cancel]
    ring

theorem sd_dot_bound (w : ℕ → α) (a : α) (ha : 0 ≤ a) (hw : ∀ t, |w t| ≤ a)
    (x : ℕ → α) (n : ℕ) (hn : 1 ≤ n) :
    |∑ t ∈ range n, sdR w a t * x t|
      ≤ a * (|x (n - 1)| + ∑ t ∈ Ico 1 n, |x t - x (t - 1)|) := by
  rw [sd_abel w a x n hn]
  have hb := sd_state_bound w a ha hw
  have h1 : |sdState w a n * x (n - 1)| ≤ a * |x (n - 1)| := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_right (hb n) (abs_nonneg _)
  have h2 : |∑ t ∈ Ico 1 n, sdState w a t * (x t - x (t - 1))|
      ≤ a * ∑ t ∈ Ico 1 n, |x t - x (t - 1)| := by
    refine (abs_sum_le_sum_abs _ _).trans ?_
    rw [mul_sum]
    refine sum_le_sum fun t _ => ?_
    rw [abs_mul]; exact mul_le_mul_of_nonneg_right (hb t) (abs_nonneg _)
  calc _ ≤ |sdState w a n * x (n - 1)|
          + |∑ t ∈ Ico 1 n, sdState w a t * (x t - x (t - 1))| := abs_sub _ _
    _ ≤ _ := by rw [mul_add]; linarith

end Core

section Shaping

variable {α : Type*} [CommRing α] {m : ℕ}

theorem shape_dot (U : Matrix (Fin m) (Fin m) α) (ε x : Fin m → α) :
    (U *ᵥ ε) ⬝ᵥ x = ε ⬝ᵥ (Uᵀ *ᵥ x) := by
  rw [dotProduct_comm, Matrix.dotProduct_mulVec, Matrix.mulVec_transpose, dotProduct_comm]

theorem shape_null_exact (U : Matrix (Fin m) (Fin m) α) (ε x : Fin m → α)
    (h : ∀ i, ε i = 0 ∨ (Uᵀ *ᵥ x) i = 0) : (U *ᵥ ε) ⬝ᵥ x = 0 := by
  rw [shape_dot]
  refine Finset.sum_eq_zero fun i _ => ?_
  rcases h i with h | h <;> simp [h]

end Shaping

section Witness

/-- Plain sign rounding residual. -/
def plainR (w : ℕ → ℚ) (a : ℚ) (t : ℕ) : ℚ := w t - a * (if 0 ≤ w t then 1 else -1)

/-- Over `ℚ`, `w ≡ 1/3`, `a = 1`, `x ≡ 1`, `n = 6`: sigma-delta beats plain sign rounding. -/
theorem sd_beats_sign :
    |∑ t ∈ range 6, sdR (fun _ => (1 : ℚ) / 3) 1 t * (fun _ => (1 : ℚ)) t|
      < |∑ t ∈ range 6, plainR (fun _ => (1 : ℚ) / 3) 1 t * (fun _ => (1 : ℚ)) t| := by
  have h := sd_prefix (fun _ => (1 : ℚ) / 3) 1 6
  simp only [mul_one]
  rw [h]
  simp [sdState, sdSignAt, plainR]
  norm_num

end Witness

end Numeric.SigmaDelta
