import Apeiron.Combinator.Basic
import Nous.Combinator.Church
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Rat.Lemmas
import Mathlib.Tactic

/-!
# Double Cover: Spin(2) → SO(2) and the Squaring Map

Combinatory logic squaring `W mul`, the double cover $U(1) \to U(1)$ via $z \mapsto z^2$
with two-to-one fibers, the exact spinor-to-rotation rational coordinates, the Gauss
three-multiplication complex product, and explicit witness triples.

Content:
1. Combinator: `W_squares` and semantic reading `W_square_sem`.
2. Double cover: `sq_fiber`, `sq_surjective_circle`, `sq_two_to_one`, `sq_surjective_unit`,
   `sq_two_to_one_unit`.
3. Spinor-to-rotation: `pythagorean_rotation`, `pythagorean_rotation_rat`,
   `pythagorean_rotation_real`, `spinor_rotation_rat`, `spinor_rotation_components`,
   `half_angle`, `half_angle_eq_spinor`.
4. Composition cost: `gauss_three_mult` (3 real products instead of 4).
5. Witnesses: `#guard` and verified example instances.
-/

set_option linter.apeiron.notation false

namespace Nous.Combinator.DoubleCover

open Apeiron.Combinator
open Apeiron.Combinator.Comb
open Nous.Combinator
open Complex Circle

/-! ### 1. Combinator: Squaring as `W mul` -/

/-- `W_squares`: `W m x ↠ m x x` specialized from `W_spec`. -/
theorem W_squares (m x : Comb) :
    Reduces (app (app W m) x) (app (app m x) x) :=
  W_spec m x

/-- Semantic reading: under any first-order interpretation where `m` reads as binary
multiplication, `W m x` reads as `x * x`. In Lean's lambda calculus:
`(fun x => x * x) = fun x => (fun a b => a * b) x x`. -/
theorem W_square_sem {α : Type _} [Mul α] :
    (fun (x : α) => x * x) = fun (x : α) => (fun (a b : α) => a * b) x x := rfl

/-! ### 2. Double Cover: $z \mapsto z^2$ on $\mathbb{C}$ and the Unit Circle -/

/-- The fiber equation for complex squaring: $z^2 = w^2 \iff z = w \lor z = -w$. -/
theorem sq_fiber (z w : ℂ) : z ^ 2 = w ^ 2 ↔ z = w ∨ z = -w := by
  constructor
  · intro h
    have hsub : z ^ 2 - w ^ 2 = 0 := sub_eq_zero.mpr h
    have hfac : (z - w) * (z + w) = 0 := by
      calc (z - w) * (z + w) = z ^ 2 - w ^ 2 := by ring
      _ = 0 := hsub
    cases mul_eq_zero.mp hfac with
    | inl h1 => left; exact sub_eq_zero.mp h1
    | inr h2 => right; exact add_eq_zero_iff_eq_neg.mp h2
  · rintro (rfl | rfl)
    · rfl
    · ring

/-- On the unit circle `Circle`, squaring is surjective. -/
theorem sq_surjective_circle (w : Circle) : ∃ z : Circle, z ^ 2 = w := by
  rcases exp_surjective w with ⟨θ, rfl⟩
  use Circle.exp (θ / 2)
  have h : Circle.exp (θ / 2) ^ 2 = Circle.exp (2 * (θ / 2)) := by
    rw [← Circle.exp_natCast_mul]
    rfl
  rw [h]
  congr 1
  ring

/-- On the unit circle `Circle`, squaring is two-to-one: the fiber has exactly two elements
`{z0, -z0}` with `z0 ≠ -z0`. This is the covering degree 2 of $\mathrm{Spin}(2) \to \mathrm{SO}(2)$. -/
theorem sq_two_to_one (w : Circle) :
    ∃ z0 : Circle, z0 ^ 2 = w ∧ {z : Circle | z ^ 2 = w} = {z0, -z0} ∧ z0 ≠ -z0 := by
  rcases sq_surjective_circle w with ⟨z0, hz0⟩
  refine ⟨z0, hz0, ?_, ?_⟩
  · ext z
    simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]
    constructor
    · intro hz
      have hz_coe : (z : ℂ) ^ 2 = (w : ℂ) := by rw [← coe_pow, hz]
      have hz0_coe : (z0 : ℂ) ^ 2 = (w : ℂ) := by rw [← coe_pow, hz0]
      have heq : (z : ℂ) ^ 2 = (z0 : ℂ) ^ 2 := by rw [hz_coe, hz0_coe]
      rw [sq_fiber] at heq
      rcases heq with h | h
      · left; exact Circle.ext h
      · right
        apply Circle.ext
        rw [coe_neg, h]
    · rintro (rfl | rfl)
      · exact hz0
      · rw [← Circle.coe_inj, coe_pow, coe_neg, neg_sq, ← coe_pow, hz0]
  · intro h
    have hne := Circle.neg_ne_self z0
    exact hne h.symm

/-- On unit complex numbers $\{z \in \mathbb{C} \mid \|z\| = 1\}$, squaring is surjective. -/
theorem sq_surjective_unit (w : ℂ) (hw : ‖w‖ = 1) : ∃ z : ℂ, ‖z‖ = 1 ∧ z ^ 2 = w := by
  let wc : Circle := ⟨w, mem_sphere_zero_iff_norm.2 hw⟩
  rcases sq_surjective_circle wc with ⟨z, hz⟩
  refine ⟨(z : ℂ), z.norm_coe, ?_⟩
  exact congrArg Subtype.val hz

/-- On unit complex numbers $\{z \in \mathbb{C} \mid \|z\| = 1\}$, squaring is two-to-one:
the fiber $\{z \in \mathbb{C} \mid \|z\| = 1 \wedge z^2 = w\}$ is $\{z_0, -z_0\}$ with $z_0 \ne -z_0$. -/
theorem sq_two_to_one_unit (w : ℂ) (hw : ‖w‖ = 1) :
    ∃ z0 : ℂ, ‖z0‖ = 1 ∧ z0 ^ 2 = w ∧ {z : ℂ | ‖z‖ = 1 ∧ z ^ 2 = w} = {z0, -z0} ∧ z0 ≠ -z0 := by
  let wc : Circle := ⟨w, mem_sphere_zero_iff_norm.2 hw⟩
  rcases sq_two_to_one wc with ⟨z0c, hz0c, _, hne⟩
  refine ⟨(z0c : ℂ), z0c.norm_coe, congrArg Subtype.val hz0c, ?_, ?_⟩
  · ext z
    simp only [Set.mem_setOf_eq, Set.mem_insert_iff, Set.mem_singleton_iff]
    constructor
    · intro ⟨_, hz_sq⟩
      have hz_sq' : z ^ 2 = (z0c : ℂ) ^ 2 := by
        rw [hz_sq, ← coe_pow, hz0c]
      rw [sq_fiber] at hz_sq'
      exact hz_sq'
    · rintro (rfl | rfl)
      · refine ⟨z0c.norm_coe, congrArg Subtype.val hz0c⟩
      · refine ⟨by simp [norm_neg, z0c.norm_coe], ?_⟩
        rw [neg_sq]
        exact congrArg Subtype.val hz0c
  · intro h
    have h_contra : z0c = -z0c := Circle.ext (by rw [coe_neg]; exact h)
    exact hne h_contra

/-! ### 3. Spinor-to-Rotation: Exact Pythagorean Coordinates -/

/-- The classical integer Pythagorean parametrization identity from squaring $a + bi$:
$(a^2 - b^2)^2 + (2ab)^2 = (a^2 + b^2)^2$. -/
theorem pythagorean_rotation (a b : ℤ) :
    (a ^ 2 - b ^ 2) ^ 2 + (2 * a * b) ^ 2 = (a ^ 2 + b ^ 2) ^ 2 := by
  ring

/-- Pythagorean rotation identity over rational numbers $\mathbb{Q}$. -/
theorem pythagorean_rotation_rat (a b : ℚ) :
    (a ^ 2 - b ^ 2) ^ 2 + (2 * a * b) ^ 2 = (a ^ 2 + b ^ 2) ^ 2 := by
  ring

/-- Pythagorean rotation identity over real numbers $\mathbb{R}$. -/
theorem pythagorean_rotation_real (a b : ℝ) :
    (a ^ 2 - b ^ 2) ^ 2 + (2 * a * b) ^ 2 = (a ^ 2 + b ^ 2) ^ 2 := by
  ring

/-- Rational rotation unit sum: $\cos^2 + \sin^2 = 1$ over $\mathbb{Q}$. -/
theorem spinor_rotation_rat (a b : ℚ) (h : a ^ 2 + b ^ 2 ≠ 0) :
    let cos := (a ^ 2 - b ^ 2) / (a ^ 2 + b ^ 2)
    let sin := (2 * a * b) / (a ^ 2 + b ^ 2)
    cos ^ 2 + sin ^ 2 = 1 := by
  intro cos sin
  dsimp [cos, sin]
  have hpyth := pythagorean_rotation_rat a b
  have hdiv : ((a ^ 2 - b ^ 2) ^ 2 + (2 * a * b) ^ 2) / (a ^ 2 + b ^ 2) ^ 2 = 1 := by
    rw [hpyth, div_self]
    exact pow_ne_zero 2 h
  rw [div_pow, div_pow, ← add_div]
  exact hdiv

/-- The normalized rotation $s^2 / |s|^2$ for $s = a + bi$ has exact components:
$\mathrm{Re}(s^2/|s|^2) = (a^2 - b^2)/(a^2 + b^2)$ and
$\mathrm{Im}(s^2/|s|^2) = 2ab / (a^2 + b^2)$. -/
theorem spinor_rotation_components (a b : ℝ) (h : a ^ 2 + b ^ 2 ≠ 0) :
    let s : ℂ := ⟨a, b⟩
    let rot : ℂ := s ^ 2 / (normSq s : ℂ)
    rot.re = (a ^ 2 - b ^ 2) / (a ^ 2 + b ^ 2) ∧
    rot.im = (2 * a * b) / (a ^ 2 + b ^ 2) := by
  intro s rot
  have hns : normSq s = a ^ 2 + b ^ 2 := by
    dsimp [s, normSq]
    ring
  have hs2 : s ^ 2 = ⟨a ^ 2 - b ^ 2, 2 * a * b⟩ := by
    apply Complex.ext
    · simp only [pow_two, mul_re]
      ring
    · simp only [pow_two, mul_im]
      ring
  dsimp [rot]
  rw [hs2, hns]
  have _ : a ^ 2 + b ^ 2 ≠ 0 := h
  exact ⟨div_ofReal_re ⟨a ^ 2 - b ^ 2, 2 * a * b⟩ (a ^ 2 + b ^ 2),
         div_ofReal_im ⟨a ^ 2 - b ^ 2, 2 * a * b⟩ (a ^ 2 + b ^ 2)⟩

/-- Half-angle tangent substitution $t = \tan(\theta/2)$:
$\cos = (1 - t^2)/(1 + t^2)$, $\sin = 2t/(1 + t^2)$ satisfies $\cos^2 + \sin^2 = 1$. -/
theorem half_angle (t : ℝ) :
    let cos := (1 - t ^ 2) / (1 + t ^ 2)
    let sin := (2 * t) / (1 + t ^ 2)
    cos ^ 2 + sin ^ 2 = 1 := by
  intro cos sin
  dsimp [cos, sin]
  have hpos : 1 + t ^ 2 ≠ 0 := by positivity
  have hnum : (1 - t ^ 2) ^ 2 + (2 * t) ^ 2 = (1 + t ^ 2) ^ 2 := by ring
  rw [div_pow, div_pow, ← add_div, hnum, div_self (pow_ne_zero 2 hpos)]

/-- Equivalence of the half-angle substitution $t = b / a$ with the spinor coordinates. -/
theorem half_angle_eq_spinor (a b : ℝ) (ha : a ≠ 0) :
    let t := b / a
    (1 - t ^ 2) / (1 + t ^ 2) = (a ^ 2 - b ^ 2) / (a ^ 2 + b ^ 2) ∧
    (2 * t) / (1 + t ^ 2) = (2 * a * b) / (a ^ 2 + b ^ 2) := by
  intro t
  dsimp [t]
  have ha2 : a ^ 2 ≠ 0 := pow_ne_zero 2 ha
  constructor
  · have h1 : 1 - (b / a) ^ 2 = (a ^ 2 - b ^ 2) / a ^ 2 := by
      rw [div_pow]
      field_simp
    have h2 : 1 + (b / a) ^ 2 = (a ^ 2 + b ^ 2) / a ^ 2 := by
      rw [div_pow]
      field_simp
    rw [h1, h2, div_div_div_cancel_right₀ ha2]
  · have h2 : 1 + (b / a) ^ 2 = (a ^ 2 + b ^ 2) / a ^ 2 := by
      rw [div_pow]
      field_simp
    rw [h2]
    have h1 : 2 * (b / a) = (2 * a * b) / a ^ 2 := by
      calc 2 * (b / a) = (2 * b * a) / (a * a) := by
            rw [mul_div_mul_right (2 * b) a ha]
            ring
      _ = (2 * a * b) / a ^ 2 := by
            congr 1
            · ring
            · ring
    rw [h1, div_div_div_cancel_right₀ ha2]

/-! ### 4. Composition Cost: Gauss 3-Multiplication Algorithm

Gauss observed that $(a + bi)(c + di)$ can be computed using 3 real multiplications
instead of 4:
  $k_1 = c(a + b)$
  $k_2 = a(d - c)$
  $k_3 = b(c + d)$
Then:
  $\mathrm{Re} = k_1 - k_3 = ac - bd$
  $\mathrm{Im} = k_1 + k_2 = ad + bc$

Compare this 3-multiply form with `Substrate.Algebra.NullCell.SpinorFold.nmul_is_diagonal`,
which costs only two real products in the null/hyperbolic basis for $\sigma = 1$. -/

theorem gauss_three_mult (a b c d : ℝ) :
    let k1 := c * (a + b)
    let k2 := a * (d - c)
    let k3 := b * (c + d)
    (⟨a, b⟩ : ℂ) * ⟨c, d⟩ = ⟨k1 - k3, k1 + k2⟩ := by
  intro k1 k2 k3
  apply Complex.ext
  · simp only [mul_re, k1, k3]
    ring
  · simp only [mul_im, k1, k2]
    ring

/-- The ring-generic algebraic identity underlying the Gauss 3-multiplication trick. -/
theorem gauss_three_mult_ring {R : Type _} [CommRing R] (a b c d : R) :
    let k1 := c * (a + b)
    let k2 := a * (d - c)
    let k3 := b * (c + d)
    (a * c - b * d = k1 - k3) ∧ (b * c + a * d = k1 + k2) := by
  intro k1 k2 k3
  dsimp [k1, k2, k3]
  constructor <;> ring

/-! ### 5. Witnesses: Pythagorean Triples and Fiber Equivalence -/

-- Witness 1: s = 2 + i gives the triple (3, 4, 5)
#guard ((2 : ℤ) ^ 2 - 1 ^ 2 = 3)
#guard (2 * (2 : ℤ) * 1 = 4)
#guard ((2 : ℤ) ^ 2 + 1 ^ 2 = 5)
#guard ((3 : ℤ) ^ 2 + 4 ^ 2 = 5 ^ 2)

example :
    let a : ℤ := 2
    let b : ℤ := 1
    (a ^ 2 - b ^ 2 = 3) ∧ (2 * a * b = 4) ∧ (a ^ 2 + b ^ 2 = 5) ∧
    (3 : ℤ) ^ 2 + 4 ^ 2 = 5 ^ 2 := by
  decide

-- Witness 2: s = 1 + 2i and -1 - 2i give the exact same rotation (-3/5, 4/5)
noncomputable def spinorRot (s : ℂ) : ℂ := s ^ 2 / (normSq s : ℂ)

example :
    let s1 : ℂ := ⟨1, 2⟩
    let s2 : ℂ := ⟨-1, -2⟩
    s2 = -s1 ∧ spinorRot s1 = spinorRot s2 := by
  intro s1 s2
  have hneg : s2 = -s1 := by
    apply Complex.ext <;> norm_num [s1, s2]
  refine ⟨hneg, ?_⟩
  dsimp [spinorRot]
  rw [hneg, neg_sq]
  congr 1
  congr 1
  exact (normSq_neg s1).symm

example :
    let s1 : ℂ := ⟨1, 2⟩
    (spinorRot s1).re = -3 / 5 ∧ (spinorRot s1).im = 4 / 5 := by
  have hne : (1 : ℝ) ^ 2 + 2 ^ 2 ≠ 0 := by norm_num
  have h := spinor_rotation_components 1 2 hne
  dsimp [spinorRot]
  constructor
  · have h1 := h.1
    norm_num at h1 ⊢
    exact h1
  · have h2 := h.2
    norm_num at h2 ⊢
    exact h2

end Nous.Combinator.DoubleCover
