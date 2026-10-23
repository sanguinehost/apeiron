import Mathlib
import ApeironWord
import Learn.Grad.DenseNorm

/-!
# Phantom gradients: a zero executed loss with a nonzero tape gradient

Source: E. Litman, "How to run Turing-complete malware in PyTorch's backward pass",
`https://elonlit.com/scrivings/how-to-run-turing-complete-malware-in-pytorchs-backward-pass/`,
retrieved to `/home/socol/oc-tmp/papers/backward-pass-malware/page.txt`. The article builds
`h(u) = (2^30 + u) - 2^30`, which float32 rounds to `0` for small `u` while autograd assigns it
derivative `1`, and a loss `(v - h v) * h θ` whose executed value is `0` and whose gradient is `v`.
SGD at learning rate 1 then writes an arbitrary update `θ ← θ - v` into the weights.

Model. `Expr` is a one-variable arithmetic expression. `Expr.eval A` executes it in an arithmetic
`A` that stores each operation's result (possibly rounded). `Expr.tape A` is what a tape computes:
the real chain rule for each recorded operation, evaluated at the executed operand values; the
product rule `d(ab) = da·b + a·db` is the dual-number product over `R[x]/(x²)` (σ = 0; Clifford,
1873) with the real part rounded and the `x`-part not.

* §A `trojan_phantom`: under round-to-128 near `2^30` (a model of float32 there), the article's loss
  evaluates to `0` on `|θ| < 64` and its tape gradient is `v`.
* §B `exact_no_phantom`, `exact_no_phantom_window`: in exact arithmetic the tape gradient is the
  derivative of the executed polynomial, so a loss that is `0` on more than `deg` points has tape
  gradient `0` everywhere. `trojan_exact`: the article's loss in exact arithmetic is `0` with
  gradient `0`.
* §C `checked_shift_exact`: the article's `(C + u) - C` in checked `i64` words is `u` or a refusal,
  never `0` for `u ≠ 0`.
* §D `q32_mul_phantom`: the boundary. A Q32.32 rounded multiply hosts a phantom
  (`(u · 2^-16) · 2^16` executes to `0` for `|u| < 2^-17`, tape derivative `1`). Exact add/sub alone
  do not exclude phantoms from the crate; the crate's exclusion is structural (one seed `q - p` from
  executed values; zero seed gives zero gradient), graded `tested` by
  `src/learn/grad/model/tests.rs` `zero_discrepancy_gives_zero_gradient`.
* §E `extracted_mul_round_phantom`: the same boundary on the shipped body. The extracted
  `learn_port.dense.mul_round` (`src/learn/grad/dense.rs:65`, the backward pass's Q32.32 multiply)
  returns `some 0` for the word `b = 2^16` (real `2^-16`) and every `|u| < 2^15`, while the exact
  product `u·b` is nonzero for `u ≠ 0`.

No `sorry`, no `axiom`, no `maxHeartbeats`; each headline has an in-module `#print axioms` row.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.Security.PhantomGradient

open Polynomial

/-- A one-variable arithmetic expression over `ℚ`. -/
inductive Expr
  | const (c : ℚ)
  | var
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)

/-- An executed arithmetic: the value each operation stores. -/
structure Arith where
  add : ℚ → ℚ → ℚ
  sub : ℚ → ℚ → ℚ
  mul : ℚ → ℚ → ℚ

/-- Executed value. O(size) operations, structural recursion. -/
def Expr.eval (A : Arith) (x : ℚ) : Expr → ℚ
  | const c => c
  | var => x
  | add a b => A.add (a.eval A x) (b.eval A x)
  | sub a b => A.sub (a.eval A x) (b.eval A x)
  | mul a b => A.mul (a.eval A x) (b.eval A x)

/-- Tape gradient: the real chain rule per recorded operation at the executed operand values.
O(size²) evaluations as written (each `mul` re-evaluates its operands); a tape caches them. -/
def Expr.tape (A : Arith) (x : ℚ) : Expr → ℚ
  | const _ => 0
  | var => 1
  | add a b => a.tape A x + b.tape A x
  | sub a b => a.tape A x - b.tape A x
  | mul a b => a.tape A x * b.eval A x + a.eval A x * b.tape A x

/-- Exact arithmetic. -/
def exact : Arith := ⟨(· + ·), (· - ·), (· * ·)⟩

@[simp] theorem exact_add (a b : ℚ) : exact.add a b = a + b := rfl
@[simp] theorem exact_sub (a b : ℚ) : exact.sub a b = a - b := rfl
@[simp] theorem exact_mul (a b : ℚ) : exact.mul a b = a * b := rfl

/-- Round to the nearest multiple of `g`, ties up. -/
def rnd (g q : ℚ) : ℚ := g * ⌊q / g + 1 / 2⌋

/-- The article's phantom `h(u) = (2^30 + u) - 2^30`. -/
def phantom (u : Expr) : Expr := .sub (.add (.const (2 ^ 30)) u) (.const (2 ^ 30))

/-- The article's loss `(v - h v) · h θ`, with `v` detached data and `θ` the variable. -/
def trojan (v : ℚ) : Expr := .mul (.sub (.const v) (phantom (.const v))) (phantom .var)

/-! ## §A — the phantom under rounding -/

/-- Model of float32 near `2^30`: an add or sub result of magnitude `≥ 2^29` is rounded to the
nearest multiple of `128`; smaller results and products are exact. -/
def fp30 : Arith where
  add a b := if (2 : ℚ) ^ 29 ≤ |a + b| then rnd 128 (a + b) else a + b
  sub a b := if (2 : ℚ) ^ 29 ≤ |a - b| then rnd 128 (a - b) else a - b
  mul a b := a * b

theorem fp30_add_big (a b : ℚ) (h : (2 : ℚ) ^ 29 ≤ |a + b|) : fp30.add a b = rnd 128 (a + b) :=
  if_pos h

theorem fp30_sub_small (a b : ℚ) (h : |a - b| < 2 ^ 29) : fp30.sub a b = a - b :=
  if_neg (not_le.mpr h)

@[simp] theorem fp30_mul (a b : ℚ) : fp30.mul a b = a * b := rfl

theorem rnd_shift (u : ℚ) (hu : |u| < 64) : rnd 128 (2 ^ 30 + u) = 2 ^ 30 := by
  rw [abs_lt] at hu
  have h : ⌊(2 ^ 30 + u) / 128 + 1 / 2⌋ = (2 ^ 23 : ℤ) := by
    rw [Int.floor_eq_iff]
    push_cast
    constructor <;> linarith
  rw [rnd, h]
  norm_num

theorem phantom_eval_fp30 (e : Expr) (x : ℚ) (hu : |e.eval fp30 x| < 64) :
    (phantom e).eval fp30 x = 0 := by
  have hlo : (2 : ℚ) ^ 29 ≤ |2 ^ 30 + e.eval fp30 x| := by
    rw [abs_lt] at hu
    rw [le_abs]
    left
    linarith
  show fp30.sub (fp30.add (2 ^ 30) (e.eval fp30 x)) (2 ^ 30) = 0
  rw [fp30_add_big _ _ hlo, rnd_shift _ hu, fp30_sub_small _ _ (by norm_num), sub_self]

/-- The tape sees the phantom as the identity, in every arithmetic. -/
theorem phantom_tape (A : Arith) (e : Expr) (x : ℚ) : (phantom e).tape A x = e.tape A x := by
  simp [phantom, Expr.tape]

/-- **The article's exploit in the model.** Executed loss `0`; tape gradient `v`. -/
theorem trojan_phantom (v θ : ℚ) (hv : |v| < 64) (hθ : |θ| < 64) :
    (trojan v).eval fp30 θ = 0 ∧ (trojan v).tape fp30 θ = v := by
  have hθ0 : (phantom .var).eval fp30 θ = 0 := phantom_eval_fp30 .var θ hθ
  have hv0 : (phantom (.const v)).eval fp30 θ = 0 := phantom_eval_fp30 (.const v) θ hv
  have hsmall : |v - 0| < (2 : ℚ) ^ 29 := by
    rw [sub_zero]
    linarith
  constructor
  · simp only [trojan, Expr.eval, hθ0, fp30_mul, mul_zero]
  · simp only [trojan, Expr.tape, phantom_tape, Expr.eval, hθ0, hv0,
      fp30_sub_small _ _ hsmall]
    simp

/-! ## §B — no phantom in exact arithmetic -/

/-- The executed polynomial. -/
noncomputable def Expr.poly : Expr → ℚ[X]
  | const c => C c
  | var => X
  | add a b => a.poly + b.poly
  | sub a b => a.poly - b.poly
  | mul a b => a.poly * b.poly

/-- Degree bound. -/
def Expr.deg : Expr → ℕ
  | const _ => 0
  | var => 1
  | add a b => max a.deg b.deg
  | sub a b => max a.deg b.deg
  | mul a b => a.deg + b.deg

theorem eval_exact (e : Expr) (x : ℚ) : e.eval exact x = e.poly.eval x := by
  induction e with
  | const c => simp [Expr.eval, Expr.poly]
  | var => simp [Expr.eval, Expr.poly]
  | add a b ha hb => simp [Expr.eval, Expr.poly, ha, hb]
  | sub a b ha hb => simp [Expr.eval, Expr.poly, ha, hb]
  | mul a b ha hb => simp [Expr.eval, Expr.poly, ha, hb]

/-- In exact arithmetic the tape gradient is the derivative of the executed polynomial. -/
theorem tape_exact (e : Expr) (x : ℚ) : e.tape exact x = (derivative e.poly).eval x := by
  induction e with
  | const c => simp [Expr.tape, Expr.poly]
  | var => simp [Expr.tape, Expr.poly]
  | add a b ha hb => simp [Expr.tape, Expr.poly, ha, hb]
  | sub a b ha hb => simp [Expr.tape, Expr.poly, ha, hb]
  | mul a b ha hb =>
    simp only [Expr.tape, Expr.poly, ha, hb, eval_exact, derivative_mul, eval_add, eval_mul]

theorem natDegree_poly_le (e : Expr) : e.poly.natDegree ≤ e.deg := by
  induction e with
  | const c => simp [Expr.poly, Expr.deg]
  | var => simp [Expr.poly, Expr.deg]
  | add a b ha hb =>
    exact (natDegree_add_le _ _).trans (max_le_max ha hb)
  | sub a b ha hb =>
    exact (natDegree_sub_le _ _).trans (max_le_max ha hb)
  | mul a b ha hb =>
    exact natDegree_mul_le.trans (Nat.add_le_add ha hb)

/-- Exact arithmetic: a loss that executes to `0` everywhere has tape gradient `0` everywhere. -/
theorem exact_no_phantom (e : Expr) (h : ∀ x, e.eval exact x = 0) (x : ℚ) :
    e.tape exact x = 0 := by
  have hp : e.poly = 0 := Polynomial.funext fun y => by simpa [eval_exact] using h y
  simp [tape_exact, hp]

/-- Exact arithmetic, bounded window: a loss that executes to `0` on more than `deg` points has
tape gradient `0` everywhere. -/
theorem exact_no_phantom_window (e : Expr) (s : Finset ℚ) (hs : e.deg < s.card)
    (h : ∀ x ∈ s, e.eval exact x = 0) (x : ℚ) : e.tape exact x = 0 := by
  have hp : e.poly = 0 :=
    eq_zero_of_natDegree_lt_card_of_eval_eq_zero' _ s
      (fun y hy => by simpa [eval_exact] using h y hy)
      ((natDegree_poly_le e).trans_lt hs)
  simp [tape_exact, hp]

/-- The article's loss in exact arithmetic: executed `0`, tape gradient `0`. -/
theorem trojan_exact (v θ : ℚ) : (trojan v).eval exact θ = 0 ∧ (trojan v).tape exact θ = 0 := by
  constructor <;> simp [trojan, phantom, Expr.eval, Expr.tape, exact]

/-! ## §C — checked words -/

/-- Checked `i64` add (the crate's `checked_add`). -/
def cadd (a b : ℤ) : Option ℤ := if -2 ^ 63 ≤ a + b ∧ a + b < 2 ^ 63 then some (a + b) else none

/-- Checked `i64` sub (the crate's `checked_sub`). -/
def csub (a b : ℤ) : Option ℤ := if -2 ^ 63 ≤ a - b ∧ a - b < 2 ^ 63 then some (a - b) else none

/-- `(C + u) - C` in checked words returns `u` or refuses. -/
theorem checked_shift_exact (c u : ℤ) :
    ((cadd c u).bind fun s => csub s c) = some u ∨ ((cadd c u).bind fun s => csub s c) = none := by
  unfold cadd csub
  split_ifs <;> simp_all
  omega

/-! ## §D — the boundary: a rounded multiply hosts a phantom -/

/-- Q32.32 model: add and sub exact; a product rounded to the nearest multiple of `2^-32`. -/
def q32 : Arith where
  add a b := a + b
  sub a b := a - b
  mul a b := rnd (1 / 2 ^ 32) (a * b)

/-- `(u · 2^-16) · 2^16`: identity on reals, `0` when executed for small `u`. -/
def mulPhantom (u : Expr) : Expr := .mul (.mul u (.const (1 / 2 ^ 16))) (.const (2 ^ 16))

theorem q32_mul_phantom (θ : ℚ) (hθ : |θ| < 1 / 2 ^ 17) :
    (mulPhantom .var).eval q32 θ = 0 ∧ (mulPhantom .var).tape q32 θ = 1 := by
  rw [abs_lt] at hθ
  have h0 : rnd (1 / 2 ^ 32) (θ * (1 / 2 ^ 16)) = 0 := by
    have hf : ⌊θ * (1 / 2 ^ 16) / (1 / 2 ^ 32) + 1 / 2⌋ = (0 : ℤ) := by
      rw [Int.floor_eq_iff]
      push_cast
      constructor <;> nlinarith [show (0 : ℚ) < 1 / 2 ^ 32 by norm_num]
    rw [rnd, hf]
    simp
  have hz : rnd (1 / 2 ^ 32) ((0 : ℚ) * 2 ^ 16) = 0 := by
    have hf : ⌊(0 : ℚ) * 2 ^ 16 / (1 / 2 ^ 32) + 1 / 2⌋ = (0 : ℤ) := by
      rw [Int.floor_eq_iff]
      norm_num
    rw [rnd, hf]
    simp
  constructor
  · simp only [mulPhantom, Expr.eval, q32, h0, hz]
  · simp only [mulPhantom, Expr.tape, Expr.eval, q32, h0]
    norm_num

/-! ## §E — the boundary on the extracted backward multiply -/

open Aeneas Aeneas.Std Result apeiron_word in
/-- **The shipped multiply rounds a small product to `0`.** `learn_port.dense.mul_round u b` with
`b.val = 2^16` and `|u.val| < 2^15` returns `some z` with `z.val = 0`; the exact product `u.val·b.val`
is `0` only at `u.val = 0`. A tape that differentiated through this multiply would assign it
derivative `b`, the §D phantom on the real body. -/
theorem extracted_mul_round_phantom (u b : I64) (hb : b.val = 2 ^ 16) (hu : |u.val| < 2 ^ 15) :
    (∃ z : I64, learn_port.dense.mul_round u b = ok (some z) ∧ z.val = 0) ∧
      (u.val * b.val = 0 ↔ u.val = 0) := by
  rw [abs_lt] at hu
  have hm : apeiron_norm.mulRound u.val b.val = 0 := by
    rw [apeiron_norm.mulRound, apeiron_norm.round32, apeiron_norm.rshr, hb]
    apply Int.ediv_eq_zero_of_lt
    · have : (0 : ℤ) ≤ u.val * 2 ^ 16 + 2 ^ 31 := by nlinarith
      exact_mod_cast this
    · have : u.val * 2 ^ 16 + 2 ^ 31 < (2 : ℤ) ^ 32 := by nlinarith
      exact_mod_cast this
  refine ⟨?_, ?_⟩
  · obtain ⟨z, hz, hzv⟩ := apeiron_grad.dense_mul_round_some u b (by rw [hm]; norm_num)
    exact ⟨z, hz, hzv.trans hm⟩
  · rw [hb]
    constructor
    · intro h; exact (mul_eq_zero.mp h).resolve_right (by norm_num)
    · intro h; rw [h, zero_mul]

end Apeiron.Security.PhantomGradient

#print axioms Apeiron.Security.PhantomGradient.trojan_phantom
#print axioms Apeiron.Security.PhantomGradient.exact_no_phantom
#print axioms Apeiron.Security.PhantomGradient.exact_no_phantom_window
#print axioms Apeiron.Security.PhantomGradient.trojan_exact
#print axioms Apeiron.Security.PhantomGradient.checked_shift_exact
#print axioms Apeiron.Security.PhantomGradient.q32_mul_phantom
#print axioms Apeiron.Security.PhantomGradient.extracted_mul_round_phantom
