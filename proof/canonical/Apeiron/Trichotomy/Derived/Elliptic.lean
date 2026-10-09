import Apeiron.Trichotomy.Basic
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.Flow
import Apeiron.Trichotomy.Derived.CStar
import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# The elliptic face is the complex plane

@name Elliptic
@cite gauss-1831
@cite argand-1806
@cite euler-1748
@cite mathlib-complex
@grade proved

Below zero the size `z.1 ^ 2 - σ * z.2 ^ 2` is a sum of two squares, so no non-zero point has
size zero and the algebra is a field. Stretching the second slot by `Real.sqrt (-σ)` turns that
size into `a ^ 2 + b ^ 2` and the multiplication into complex multiplication. The elliptic face
is the complex plane, and `ellipticEquiv` is the relabelling.

The relabelling carries each operation to its complex counterpart. `ellipticEquiv_mul` sends
`mul σ` to complex multiplication, `ellipticEquiv_normSq` sends `normForm σ` to
`Complex.normSq`, `ellipticEquiv_conj` sends `conj` to complex conjugation, and
`ellipticEquiv_flow` sends the flow to `Complex.exp (t * i)`. `ellipticEquiv_affineFold`
carries a chronological affine fold to the same fold read in `ℂ`, and
`elliptic_quarter_turn_order_witness` evaluates two orders of a quarter turn at `σ = -1`,
getting `(0, 1)` against `(1, 0)`. The two orders differ, which is the order-sensitivity the old
`AttentionSpinor.spinApply` example exhibited, at the σ = -1 setting this corpus proves.

`dampedStep` weights a flow point by `ρ` and the old point by `1 - ρ`, the shape of the old
`TrainingCollapse.rotorStep`. `damped_fold_bounded` carries the one-step size bound through a
chronological fold, and `damped_step_joint_witness` evaluates one step at `σ = -4`.

The flow on the elliptic face is a rotation on the unit circle, and its discrete form is the
Cayley step. `skew_re_zero` says a skew-adjoint map has zero real part in `⟪A s, s⟫`, and
`cayley_isometry` says a point solving `φ + A φ = ψ - A ψ` for such a map keeps its length.
Both are stated over an arbitrary complex inner-product space, with no finite dimension and no
inverse of `1 + A`, matching `../sanguine-ref/Substrate/StationaryPhase/SpinorLoop.lean`'s
`skew_re_zero` and `cayley_isometry`. `cayley_isometry_of_re_zero` is the sharp local form:
only the real part at `φ + ψ` is assumed, not skew-adjointness everywhere.

## Root

| from | here |
|---|---|
| `mul`, `normForm` (`Trichotomy/Basic.lean`) | `ellipticEquiv_mul`, `ellipticEquiv_normSq` |
| `conj` (`Trichotomy/Basic.lean`) | `ellipticEquiv_conj` |
| `flow_neg`, `flow_neg_one` (`Trichotomy/Flow.lean`) | `ellipticEquiv_flow` |
| `ringAffineFold_apply` (`Trichotomy/AnyRing.lean`) | `ellipticEquiv_affineFold` |
| `ringAffine_hyperbolic_witness` | `elliptic_quarter_turn_order_witness` |
| `flow`, `ringAffineApply` (`Trichotomy/AnyRing.lean`) | `dampedStep` |
| `ringAffineFold_apply`, `cstarNormNeg_mul` | `damped_fold_bounded` |
| `flow_neg`, `ringAffineApply` | `damped_step_joint_witness` |
| `skew_re_zero` (`sanguine-ref`, `SpinorLoop.lean`) | `skew_re_zero` |
| `cayley_isometry` (`sanguine-ref`, `SpinorLoop.lean`) | `cayley_isometry` |
| the same pair, local real-part form | `cayley_isometry_of_re_zero` |

`noZeroDivisors_iff_disc_neg` is the field statement the relabelling makes explicit: it says
`σ < 0` is the face where a product is zero only when a factor is. The carrier and the two
operations are `Cell`, `mul` and `normForm`; the relabelling is what identifies them with `ℂ`.
`elliptic_quarter_turn_order_witness` rests on `flow_neg_one` as well as
`ringAffineFold_apply`, since its steps are the σ = -1 flow points. `damped_fold_bounded` rests
on `cstarNormNeg_mul` and the triangle inequality for `cstarNormNeg`, which is not stated in
`CStar.lean`: the step is read in `ℂ`, where `norm_add_le` supplies it.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

/-- The forward map of the relabelling: the second slot stretched by `√(−σ)`.

    (σ : ℝ) (z : Cell)   [the parameter and the point being relabelled]
    : ℂ                  [the complex numbers: pairs of reals where the second squares to −1]
    Real.sqrt            [the number that squares to give this]

    So: `(a, b)` becomes `a + i·√(−σ)·b`, which turns `a² − σb²` into `a² + b²`. -/
private def ellipticMap (σ : ℝ) (z : Cell) : ℂ :=
  ⟨z.1, Real.sqrt (-σ) * z.2⟩

/-- The inverse map of the relabelling: the second slot divided by `√(−σ)`.

    (σ : ℝ) (w : ℂ)      [the parameter and the complex number]
    : Cell               [a point: two real numbers written side by side, a + b·x]
    Real.sqrt            [the number that squares to give this]

    So: `a + i·b` becomes `(a, b / √(−σ))`. -/
private def ellipticMapInv (σ : ℝ) (w : ℂ) : Cell :=
  (w.re, w.im / Real.sqrt (-σ))

/-- The elliptic face relabelled as the complex plane.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    Cell                    [a point: two real numbers written side by side, a + b·x]
    ≃ₗ                      [can be relabelled into, keeping addition and stretching intact]
    ℂ                       [the complex numbers: pairs of reals where the second squares to −1]

    So: the forward map sends `(a, b)` to `a + i·√(−σ)·b`, and the inverse divides the second
    slot by `√(−σ)`. The stretch is what turns `a² − σb²` into `a² + b²`. -/
def ellipticEquiv (σ : ℝ) (hσ : σ < 0) : Cell ≃ₗ[ℝ] ℂ where
  toFun := ellipticMap σ
  invFun := ellipticMapInv σ
  left_inv z := by
    obtain ⟨a, b⟩ := z
    have hs : Real.sqrt (-σ) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr (by linarith))
    simp only [ellipticMap, ellipticMapInv, Prod.mk.injEq, true_and]
    exact mul_div_cancel_left₀ b hs
  right_inv w := by
    have hs : Real.sqrt (-σ) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr (by linarith))
    apply Complex.ext
    · rfl
    · simp only [ellipticMap, ellipticMapInv]
      rw [← mul_div_assoc, mul_div_cancel_left₀ w.im hs]
  map_add' z w := by
    apply Complex.ext
    · simp only [ellipticMap, Complex.add_re, Prod.fst_add]
    · simp only [ellipticMap, Complex.add_im, Prod.snd_add]
      ring
  map_smul' r z := by
    apply Complex.ext
    · simp only [ellipticMap, Complex.smul_re, Prod.smul_fst, smul_eq_mul, RingHom.id_apply]
    · simp only [ellipticMap, Complex.smul_im, Prod.smul_snd, smul_eq_mul, RingHom.id_apply]
      ring

/-- The relabelling written out on one point.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (z : Cell)              [the point being relabelled]
    ℂ                       [the complex numbers: pairs of reals where the second squares to −1]
    Real.sqrt               [the number that squares to give this]

    So: `ellipticEquiv` sends `(a, b)` to `⟨a, √(−σ)·b⟩`. -/
@[simp] private theorem ellipticEquiv_apply (σ : ℝ) (hσ : σ < 0) (z : Cell) :
    ellipticEquiv σ hσ z = (⟨z.1, Real.sqrt (-σ) * z.2⟩ : ℂ) := rfl

/-- The relabelling carries multiplication to complex multiplication.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (z w : Cell)            [the two points being multiplied]
    mul                     [multiply two points together, the σ way]
    Cell                    [a point: two real numbers written side by side, a + b·x]

    So: `ellipticEquiv (z · w) = ellipticEquiv z * ellipticEquiv w`. The stretch makes the
    `σ·b·d` term the complex product's `−(√(−σ)·b)(√(−σ)·d)` term. -/
theorem ellipticEquiv_mul (σ : ℝ) (hσ : σ < 0) (z w : Cell) :
    ellipticEquiv σ hσ (mul σ z w) = ellipticEquiv σ hσ z * ellipticEquiv σ hσ w := by
  have hsq : Real.sqrt (-σ) ^ 2 = -σ := Real.sq_sqrt (neg_nonneg.mpr hσ.le)
  rw [ellipticEquiv_apply, ellipticEquiv_apply, ellipticEquiv_apply]
  apply Complex.ext
  · simp only [Complex.mul_re]
    rw [show (Real.sqrt (-σ) * z.2) * (Real.sqrt (-σ) * w.2) =
      Real.sqrt (-σ) ^ 2 * (z.2 * w.2) by ring]
    rw [hsq]
    ring
  · simp only [Complex.mul_im]
    ring

/-- The relabelling carries the norm form to the complex size, squared.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (z : Cell)              [the point being measured]
    Complex.normSq          [the size of a complex number, squared]
    normForm                [the norm form: a point times its conjugate, which can be negative]
    ℝ                       [the real numbers: every number on the number line]

    So: `normSq (ellipticEquiv z) = normForm σ z`. Both sides are `a² + b²` once the second
    slot is stretched, and `normForm` is the algebraic form, not a Euclidean norm. -/
theorem ellipticEquiv_normSq (σ : ℝ) (hσ : σ < 0) (z : Cell) :
    Complex.normSq (ellipticEquiv σ hσ z) = normForm σ z := by
  have hsq : Real.sqrt (-σ) ^ 2 = -σ := Real.sq_sqrt (neg_nonneg.mpr hσ.le)
  rw [ellipticEquiv_apply, Complex.normSq_apply]
  simp only [normForm, AnyRing.ringNormForm]
  nlinarith [hsq]

/-- **The σ = -4 joint witness.**

    ℝ                       [the real numbers: every number on the number line]
    mul                     [multiply two points together, the σ way]
    normForm                [the norm form: a point times its conjugate, which can be negative]
    ∧                       [both are true]

    So: at `σ = -4` the generator `(0, 1)` maps to `2·i`, its square maps to `-4`, and
    `normForm (-4) (0, 1) = 4`. The three hold together at one setting, and the mutation control
    removes the `√(−σ)` stretch from the forward map, under which the first conjunct is false. -/
theorem ellipticEquiv_neg_four_witness :
    ellipticEquiv (-4) (by norm_num) ((0 : ℝ), (1 : ℝ)) = 2 * Complex.I ∧
    ellipticEquiv (-4) (by norm_num)
        (mul (-4) ((0 : ℝ), (1 : ℝ)) ((0 : ℝ), (1 : ℝ))) = -4 ∧
    normForm (-4) ((0 : ℝ), (1 : ℝ)) = 4 := by
  have hs : Real.sqrt (-(-4 : ℝ)) = 2 := by
    rw [show (-(-4 : ℝ)) = 2 ^ 2 by norm_num]
    exact Real.sqrt_sq (by norm_num)
  refine ⟨?_, ?_, ?_⟩
  · rw [ellipticEquiv_apply, hs]
    apply Complex.ext <;> norm_num
  · rw [ellipticEquiv_apply, hs]
    apply Complex.ext <;> norm_num
  · simp only [normForm, AnyRing.ringNormForm]
    norm_num

/-- The relabelling carries the conjugate to complex conjugation.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (z : Cell)              [the point being conjugated]
    conj                    [the same point with its second slot negated]
    ℂ                       [the complex numbers: pairs of reals where the second squares to −1]

    So: `ellipticEquiv (conj z) = star (ellipticEquiv z)`. Negating the second slot is
    conjugating the complex number. -/
theorem ellipticEquiv_conj (σ : ℝ) (hσ : σ < 0) (z : Cell) :
    ellipticEquiv σ hσ (conj z) = star (ellipticEquiv σ hσ z) := by
  rw [ellipticEquiv_apply, ellipticEquiv_apply]
  apply Complex.ext
  · simp only [Complex.star_def, Complex.conj_re]
  · simp only [Complex.star_def, Complex.conj_im]
    ring

/-- The relabelling carries the flow to the unit circle.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (t : ℝ)                 [a time]
    flow                    [the size-one points reached by turning or boosting for a time]
    Real.sqrt               [the number that squares to give this]
    ℂ                       [the complex numbers: pairs of reals where the second squares to −1]

    So: `ellipticEquiv (flow σ t) = Complex.exp (↑(√(−σ)·t) * i)`, the unit circle at angle
    `√(−σ)·t`. The sign of `σ` picks the face, and below zero the face is this circle. -/
theorem ellipticEquiv_flow (σ : ℝ) (hσ : σ < 0) (t : ℝ) :
    ellipticEquiv σ hσ (flow σ t) =
      Complex.exp (((Real.sqrt (-σ) * t : ℝ) : ℂ) * Complex.I) := by
  have hs : Real.sqrt (-σ) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr (by linarith))
  rw [flow_neg hσ, ellipticEquiv_apply]
  dsimp only
  rw [← mul_div_assoc, mul_div_cancel_left₀ _ hs]
  apply Complex.ext
  · simp only [Complex.exp_ofReal_mul_I_re]
  · simp only [Complex.exp_ofReal_mul_I_im]

/-- One affine step, read through the relabelling.

    (σ : ℝ) (hσ : σ < 0)                [the parameter, below zero, and the proof that it is]
    (step : AnyRing.RingAffineStep ℝ)    [a step over the ring: a coefficient and an injection]
    (z : Cell)                           [the point the step is applied to]

    So: relabelling after one step equals multiplying by the relabelled coefficient, adding the
    relabelled injection. This is `ellipticEquiv_mul` together with additivity. -/
private theorem ellipticEquiv_affineApply (σ : ℝ) (hσ : σ < 0)
    (step : AnyRing.RingAffineStep ℝ) (z : Cell) :
    ellipticEquiv σ hσ (AnyRing.ringAffineApply σ step z) =
      ellipticEquiv σ hσ step.coefficient * ellipticEquiv σ hσ z +
        ellipticEquiv σ hσ step.injection := by
  rw [AnyRing.ringAffineApply]
  change ellipticEquiv σ hσ (mul σ step.coefficient z + step.injection) = _
  rw [map_add, ellipticEquiv_mul]

/-- The relabelling carries a chronological left fold to the same fold in `ℂ`.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (steps : List (AnyRing.RingAffineStep ℝ)) [the steps in the order they are applied]
    List                    [a finite sequence of things, in order]
    (acc : Cell)            [the point the fold starts from]

    So: folding the steps over `acc` and relabelling the result equals folding the relabelled
    steps over the relabelled `acc`. The induction is on the list and uses the one-step law. -/
private theorem ellipticEquiv_foldl (σ : ℝ) (hσ : σ < 0)
    (steps : List (AnyRing.RingAffineStep ℝ)) (acc : Cell) :
    ellipticEquiv σ hσ
        (steps.foldl (fun z step => AnyRing.ringAffineApply σ step z) acc) =
      steps.foldl
        (fun acc step => ellipticEquiv σ hσ step.coefficient * acc +
          ellipticEquiv σ hσ step.injection)
        (ellipticEquiv σ hσ acc) := by
  induction steps generalizing acc with
  | nil => simp
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (AnyRing.ringAffineApply σ s acc), ellipticEquiv_affineApply]

/-- The relabelling carries the affine fold to the same fold read in `ℂ`.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (steps : List (AnyRing.RingAffineStep ℝ)) [the steps in the order they are applied]
    List                    [a finite sequence of things, in order]
    (z : Cell)              [the point the fold is applied to]

    So: the composed step applied once, then relabelled, equals the left fold of the relabelled
    steps. This is `ringAffineFold_apply` with the fold law above. -/
theorem ellipticEquiv_affineFold (σ : ℝ) (hσ : σ < 0)
    (steps : List (AnyRing.RingAffineStep ℝ)) (z : Cell) :
    ellipticEquiv σ hσ (AnyRing.ringAffineApply σ (AnyRing.ringAffineFold σ steps) z) =
      steps.foldl
        (fun acc step => ellipticEquiv σ hσ step.coefficient * acc +
          ellipticEquiv σ hσ step.injection)
        (ellipticEquiv σ hσ z) := by
  rw [AnyRing.ringAffineFold_apply]
  exact ellipticEquiv_foldl σ hσ steps z

/-- **A quarter turn at σ = -1 reads the fold's order.**

    Cell    [a point: two real numbers written side by side, a + b·x]
    flow    [the size-one points reached by turning or boosting for a time]
    ℝ       [the real numbers: every number on the number line]
    →       [if the left holds, so does the right]
    ∧       [both are true]
    ≠       [is not equal to]

    So: with `step θ b` the affine step whose coefficient is `flow (-1) θ` and whose injection is
    `b`, the fold `[step 0 (1,0), step (π/2) 0]` applied to the origin reports `(0,1)`, and the
    reverse order reports `(1,0)`. The two folds differ, so the carry records the order. This is
    the σ = -1 coverage instance of the old `AttentionSpinor.spinApply` quarter-turn example, not
    a general order theorem. -/
theorem elliptic_quarter_turn_order_witness :
    (let step : ℝ → Cell → AnyRing.RingAffineStep ℝ := fun θ b => ⟨flow (-1) θ, b⟩
     AnyRing.ringAffineApply (-1)
         (AnyRing.ringAffineFold (-1) [step 0 ((1 : ℝ), (0 : ℝ)),
           step (Real.pi / 2) 0]) 0 = ((0 : ℝ), (1 : ℝ)) ∧
     AnyRing.ringAffineApply (-1)
         (AnyRing.ringAffineFold (-1) [step (Real.pi / 2) 0,
           step 0 ((1 : ℝ), (0 : ℝ))]) 0 = ((1 : ℝ), (0 : ℝ)) ∧
     AnyRing.ringAffineFold (-1) [step 0 ((1 : ℝ), (0 : ℝ)), step (Real.pi / 2) 0] ≠
       AnyRing.ringAffineFold (-1) [step (Real.pi / 2) 0, step 0 ((1 : ℝ), (0 : ℝ))]) := by
  dsimp only
  have h1 : AnyRing.ringAffineApply (-1)
      (AnyRing.ringAffineFold (-1) [⟨flow (-1) 0, ((1 : ℝ), (0 : ℝ))⟩,
        ⟨flow (-1) (Real.pi / 2), 0⟩]) 0 = ((0 : ℝ), (1 : ℝ)) := by
    rw [AnyRing.ringAffineFold_apply]
    simp only [List.foldl_cons, List.foldl_nil, AnyRing.ringAffineApply, AnyRing.ringMul,
      flow_neg_one]
    norm_num [Real.cos_pi_div_two, Real.sin_pi_div_two]
  have h2 : AnyRing.ringAffineApply (-1)
      (AnyRing.ringAffineFold (-1) [⟨flow (-1) (Real.pi / 2), 0⟩,
        ⟨flow (-1) 0, ((1 : ℝ), (0 : ℝ))⟩]) 0 = ((1 : ℝ), (0 : ℝ)) := by
    rw [AnyRing.ringAffineFold_apply]
    simp only [List.foldl_cons, List.foldl_nil, AnyRing.ringAffineApply, AnyRing.ringMul,
      flow_neg_one]
    norm_num [Real.cos_pi_div_two, Real.sin_pi_div_two]
  refine ⟨h1, h2, ?_⟩
  intro h
  have h4 := congrArg (fun s => AnyRing.ringAffineApply (-1) s 0) h
  rw [h1, h2] at h4
  norm_num at h4

/-- **One damped affine step: a weighted flow point, and the old point weighted by its complement.**

    (σ : ℝ)                  [the parameter: what the adjoined direction squares to]
    (p : ℝ × ℝ × Cell)       [the weight, the flow time and the point being injected]
    ℝ × ℝ × Cell             [paired with, so a real, a real and a point side by side]
    AnyRing.RingAffineStep   [a step over the ring: a coefficient and an injection]

    So: `p.1` weights the flow point `flow σ p.2.1` and `1 - p.1` weights `p.2.2`. This is the
    shape of the old `TrainingCollapse.rotorStep`: a damping weight on a rotation, with the
    complement carrying the previous state. -/
noncomputable def dampedStep (σ : ℝ) (p : ℝ × ℝ × Cell) : AnyRing.RingAffineStep ℝ :=
  ⟨p.1 • flow σ p.2.1, (1 - p.1) • p.2.2⟩

/-- The elliptic size is the complex modulus read through the relabelling.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (x : Cell)              [the point being measured]
    cstarNormNeg            [the elliptic size: the square root of the point's norm form]
    ellipticEquiv           [the relabelling that sends the point to the complex numbers]
    ℝ                       [the real numbers: every number on the number line]

    So: `cstarNormNeg σ x = ‖ellipticEquiv x‖`. The relabelling turns the norm form into the
    complex size squared, and the square root of that square is the modulus. -/
private theorem cstarNormNeg_eq_norm (σ : ℝ) (hσ : σ < 0) (x : Cell) :
    cstarNormNeg σ x = ‖ellipticEquiv σ hσ x‖ := by
  unfold cstarNormNeg
  rw [← ellipticEquiv_normSq σ hσ x, Complex.normSq_eq_norm_sq,
    Real.sqrt_sq (norm_nonneg (ellipticEquiv σ hσ x))]

/-- The elliptic size obeys the triangle inequality.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (x y : Cell)            [the two points being added]
    cstarNormNeg            [the elliptic size: the square root of the point's norm form]
    ≤                       [is smaller than or equal to]

    So: `cstarNormNeg σ (x + y) ≤ cstarNormNeg σ x + cstarNormNeg σ y`. Read in `ℂ`, this is
    `norm_add_le`. -/
private theorem cstarNormNeg_add_le (σ : ℝ) (hσ : σ < 0) (x y : Cell) :
    cstarNormNeg σ (x + y) ≤ cstarNormNeg σ x + cstarNormNeg σ y := by
  rw [cstarNormNeg_eq_norm σ hσ (x + y), cstarNormNeg_eq_norm σ hσ x,
    cstarNormNeg_eq_norm σ hσ y, LinearEquiv.map_add (ellipticEquiv σ hσ) x y]
  exact norm_add_le _ _

/-- The elliptic size of a stretched point is the absolute stretch times the size.

    (σ : ℝ)                 [the parameter: what the adjoined direction squares to]
    (r : ℝ) (x : Cell)      [the stretch factor and the point being stretched]
    cstarNormNeg            [the elliptic size: the square root of the point's norm form]
    •                       [stretch by, a number times a point]

    So: `cstarNormNeg σ (r • x) = |r| * cstarNormNeg σ x`, because the size is the square root
    of a square, so a stretch factor comes out as its absolute value. -/
private theorem cstarNormNeg_smul (σ : ℝ) (r : ℝ) (x : Cell) :
    cstarNormNeg σ (r • x) = |r| * cstarNormNeg σ x := by
  unfold cstarNormNeg
  rw [normForm_smul σ r x, Real.sqrt_mul (sq_nonneg r) (normForm σ x), Real.sqrt_sq_eq_abs]

/-- The flow point has elliptic size one.

    (σ : ℝ) (θ : ℝ)                 [the parameter and a time]
    cstarNormNeg                    [the elliptic size: the square root of the point's norm form]
    flow                            [the size-one points reached by turning for a time]

    So: `cstarNormNeg σ (flow σ θ) = 1`, because the flow stays on the size-one locus. -/
private theorem cstarNormNeg_flow (σ : ℝ) (θ : ℝ) :
    cstarNormNeg σ (flow σ θ) = 1 := by
  unfold cstarNormNeg
  rw [normForm_flow, Real.sqrt_one]

/-- One damped step keeps a point inside the bound its injection satisfies.

    (σ : ℝ) (hσ : σ < 0)       [the parameter, below zero, and the proof that it is]
    (M : ℝ)                     [the size bound the point must stay inside]
    (ρ θ : ℝ)                   [the damping weight and the flow time]
    (u z : Cell)                [the injection point and the point the step is applied to]
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) [the damping weight lies between zero and one]
    (hz : cstarNormNeg σ z ≤ M) [the point starts inside the bound]
    (hu : cstarNormNeg σ u ≤ M) [the injection lies inside the bound]
    cstarNormNeg                [the elliptic size: the square root of the point's norm form]
    •                           [stretch by, a number times a point]
    flow                        [the size-one points reached by turning for a time]
    ≤                           [is smaller than or equal to]

    So: multiplying by `ρ • flow σ θ` and adding `(1 - ρ) • u` sends a point of size at most `M`
    to a point of size at most `M`. The coefficient has size `ρ`, the injection has size
    `(1 - ρ)` times its own size, and the triangle inequality adds them. -/
private theorem damped_step_norm_le (σ : ℝ) (hσ : σ < 0) (M : ℝ)
    (ρ θ : ℝ) (u z : Cell) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hz : cstarNormNeg σ z ≤ M) (hu : cstarNormNeg σ u ≤ M) :
    cstarNormNeg σ (AnyRing.ringAffineApply σ ⟨ρ • flow σ θ, (1 - ρ) • u⟩ z) ≤ M := by
  change cstarNormNeg σ (AnyRing.ringMul σ (ρ • flow σ θ) z + (1 - ρ) • u) ≤ M
  have hcoef : cstarNormNeg σ (ρ • flow σ θ) = ρ := by
    rw [cstarNormNeg_smul σ, cstarNormNeg_flow σ θ, mul_one, abs_of_nonneg hρ0]
  have hinj : cstarNormNeg σ ((1 - ρ) • u) = (1 - ρ) * cstarNormNeg σ u := by
    rw [cstarNormNeg_smul σ, abs_of_nonneg (sub_nonneg.mpr hρ1)]
  calc
    cstarNormNeg σ (AnyRing.ringMul σ (ρ • flow σ θ) z + (1 - ρ) • u)
        ≤ cstarNormNeg σ (AnyRing.ringMul σ (ρ • flow σ θ) z) +
          cstarNormNeg σ ((1 - ρ) • u) :=
          cstarNormNeg_add_le σ hσ (AnyRing.ringMul σ (ρ • flow σ θ) z) ((1 - ρ) • u)
    _ = ρ * cstarNormNeg σ z + (1 - ρ) * cstarNormNeg σ u := by
        rw [cstarNormNeg_mul hσ.le, hcoef, hinj]
    _ ≤ ρ * M + (1 - ρ) * M :=
        add_le_add (mul_le_mul_of_nonneg_left hz hρ0)
          (mul_le_mul_of_nonneg_left hu (sub_nonneg.mpr hρ1))
    _ = M := by ring

/-- The damped fold keeps a point inside the bound.

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (M : ℝ)                 [the size bound the point must stay inside]
    (ps : List (ℝ × ℝ × Cell)) [the steps, in the order they are applied]
    List                    [a finite sequence of things, in order]
    ℝ × ℝ × Cell            [paired with, so a real, a real and a point side by side]
    (hps : ∀ p ∈ ps, ...)   [every step's weight lies between zero and one, and its injection is]
    ∀                       [for every single one]
    ∈                       [is one of the things in]
    ∧                       [both are true]
    (z : Cell)              [the point the fold starts from]
    ≤                       [is smaller than or equal to]

    So: applying the steps one after another keeps the point inside `M`, by induction on the
    list with `damped_step_norm_le` at each cons. -/
private theorem damped_fold_aux (σ : ℝ) (hσ : σ < 0) (M : ℝ)
    (ps : List (ℝ × ℝ × Cell))
    (hps : ∀ p ∈ ps, 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ cstarNormNeg σ p.2.2 ≤ M)
    (z : Cell) (hz : cstarNormNeg σ z ≤ M) :
    cstarNormNeg σ
      (ps.foldl (fun acc p => AnyRing.ringAffineApply σ (dampedStep σ p) acc) z) ≤ M := by
  induction ps generalizing z with
  | nil => exact hz
  | cons p rest ih =>
    simp only [List.foldl_cons]
    have hp := hps p (by simp)
    refine ih (fun q hq => hps q (List.mem_cons_of_mem p hq))
      (AnyRing.ringAffineApply σ (dampedStep σ p) z) ?_
    exact damped_step_norm_le σ hσ M p.1 p.2.1 p.2.2 z hp.1 hp.2.1 hz hp.2.2

/-- **A chronological fold of damped steps keeps the point inside the bound.**

    (σ : ℝ) (hσ : σ < 0)   [the parameter, below zero, and the proof that it is]
    (M : ℝ)                 [the size bound the point must stay inside]
    (z : Cell)              [the point the fold starts from]
    (hz : cstarNormNeg σ z ≤ M) [the point starts inside the bound]
    (ps : List (ℝ × ℝ × Cell)) [the steps, in the order they are applied]
    List                    [a finite sequence of things, in order]
    ℝ × ℝ × Cell            [paired with, so a real, a real and a point side by side]
    (hps : ∀ p ∈ ps, ...)   [every step's weight lies between zero and one, and its injection is]
    ∀                       [for every single one]
    ∈                       [is one of the things in]
    ∧                       [both are true]
    ≤                       [is smaller than or equal to]

    So: a point of size at most `M`, folded through steps whose weights lie in `[0, 1]` and whose
    injections have size at most `M`, stays of size at most `M`. The fold is chronological, so
    the order of `ps` is the order the steps are applied. -/
theorem damped_fold_bounded (σ : ℝ) (hσ : σ < 0) (M : ℝ) (z : Cell)
    (hz : cstarNormNeg σ z ≤ M) (ps : List (ℝ × ℝ × Cell))
    (hps : ∀ p ∈ ps, 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ cstarNormNeg σ p.2.2 ≤ M) :
    cstarNormNeg σ
      (AnyRing.ringAffineApply σ (AnyRing.ringAffineFold σ (ps.map (dampedStep σ))) z) ≤ M := by
  rw [AnyRing.ringAffineFold_apply, List.foldl_map]
  exact damped_fold_aux σ hσ M ps hps z hz

/-- **The σ = -4 joint witness: one damped step, its next state, and its two sizes.**

    ℝ                       [the real numbers: every number on the number line]
    normForm                [the norm form: a point times its conjugate, possibly negative]
    ≤                       [is smaller than or equal to]
    ∧                       [both are true]

    So: at `σ = -4`, with weight `1/2`, flow time `π/4` and injection `(1, 0)`, one damped step
    sends `(1, 0)` to `(1/2, 1/4)`, that point has `normForm = 1/2`, and its elliptic size is at
    most one. The three hold together at one setting. -/
theorem damped_step_joint_witness :
    AnyRing.ringAffineApply (-4)
        (dampedStep (-4) ((1 / 2 : ℝ), (Real.pi / 4, ((1 : ℝ), (0 : ℝ)))))
        ((1 : ℝ), (0 : ℝ)) = ((1 / 2 : ℝ), (1 / 4 : ℝ)) ∧
    normForm (-4) ((1 / 2 : ℝ), (1 / 4 : ℝ)) = 1 / 2 ∧
    cstarNormNeg (-4) ((1 / 2 : ℝ), (1 / 4 : ℝ)) ≤ 1 := by
  have hs : Real.sqrt (-(-4 : ℝ)) = 2 := by
    rw [show (-(-4 : ℝ)) = 2 ^ 2 by norm_num]
    exact Real.sqrt_sq (by norm_num)
  have hflow : flow (-4) (Real.pi / 4) = ((0 : ℝ), (1 / 2 : ℝ)) := by
    rw [flow_neg (by norm_num : (-4 : ℝ) < 0), hs]
    have harg : 2 * (Real.pi / 4) = Real.pi / 2 := by ring
    rw [harg]
    norm_num [Real.cos_pi_div_two, Real.sin_pi_div_two]
  refine ⟨?_, ?_, ?_⟩
  · rw [AnyRing.ringAffineApply]
    simp only [dampedStep]
    rw [hflow]
    apply Prod.ext <;>
      norm_num [AnyRing.ringMul, Prod.smul_fst, Prod.smul_snd, smul_eq_mul,
        Prod.fst_add, Prod.snd_add, Prod.fst_mul, Prod.snd_mul]
  · simp only [normForm, AnyRing.ringNormForm]
    norm_num
  · unfold cstarNormNeg
    rw [show normForm (-4) ((1 / 2 : ℝ), (1 / 4 : ℝ)) = 1 / 2 by
      simp only [normForm, AnyRing.ringNormForm]
      norm_num]
    simpa using Real.sqrt_le_sqrt (show (1 / 2 : ℝ) ≤ 1 by norm_num)

-- ═══════════════════════════════════════════════════════════════════
-- §7. THE CAYLEY STEP ON A COMPLEX INNER-PRODUCT SPACE
-- ═══════════════════════════════════════════════════════════════════
-- The elliptic flow `ellipticEquiv_flow` is a rotation on the unit circle. Its discrete form is
-- the Cayley step: solving `φ + A φ = ψ - A ψ` for a skew-adjoint `A`. Nothing below uses the
-- finite-dimensional lattice the predecessor wrote it on, so the statements are made over an
-- arbitrary complex inner-product space.

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

/-- A skew-adjoint map has no radial part: the pairing of `A s` with `s` has zero real part.

    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]  [a complex inner-product space]
    (A : E →ₗ[ℂ] E)          [a complex-linear map of the space to itself]
    (hskew : ∀ x y, ⟪A x, y⟫_ℂ = -⟪x, A y⟫_ℂ)
                             [the map is skew-adjoint: pairing its image with a point negates]
                             [the pairing of the point with the image]
    ∀                        [for every single one]
    (s : E)                  [a point]
    RCLike.re ⟪A s, s⟫_ℂ = 0 [the pairing of the image with the point has zero real part]

    So: `inner_re_symm` swaps the two arguments of `⟪A s, s⟫` and `hskew` at `x = y = s`
    negates, so `re⟪A s, s⟫ = -re⟪A s, s⟫`, and the real part is zero. This is the predecessor's
    `skew_re_zero`, over an arbitrary complex inner-product space. -/
theorem skew_re_zero (A : E →ₗ[ℂ] E)
    (hskew : ∀ x y, ⟪A x, y⟫_ℂ = -⟪x, A y⟫_ℂ) (s : E) :
    RCLike.re ⟪A s, s⟫_ℂ = 0 := by
  have hsymm : RCLike.re ⟪s, A s⟫_ℂ = RCLike.re ⟪A s, s⟫_ℂ := inner_re_symm s (A s)
  have hneg : RCLike.re ⟪A s, s⟫_ℂ = -RCLike.re ⟪s, A s⟫_ℂ := by
    rw [hskew s s, map_neg]
  linarith

/-- **The Cayley step preserves the length, under the local real-part condition.**

    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]  [a complex inner-product space]
    (A : E →ₗ[ℂ] E)          [a complex-linear map of the space to itself]
    (φ ψ : E)                [the point before the step and the point after it]
    (hrel : φ + A φ = ψ - A ψ)
                             [the Cayley relation: the step solves this for the two points]
    (hlocal : RCLike.re ⟪A (φ + ψ), φ + ψ⟫_ℂ = 0)
                             [the sharp local premise: only the real part at `φ + ψ` is zero]
    ‖φ‖ = ‖ψ‖                [the two points have the same length]

    So: with `s = φ + ψ` and `d = φ - ψ`, the relation gives `d = -A s`, and
    `‖φ‖² - ‖ψ‖² = re⟪d, s⟫ = -re⟪A s, s⟫ = 0`. The premise is local: it asks nothing of `A`
    away from `φ + ψ`, so it is weaker than `hskew` and needs no `1 + A` inverse and no finite
    dimension. -/
theorem cayley_isometry_of_re_zero (A : E →ₗ[ℂ] E) (φ ψ : E)
    (hrel : φ + A φ = ψ - A ψ) (hlocal : RCLike.re ⟪A (φ + ψ), φ + ψ⟫_ℂ = 0) :
    ‖φ‖ = ‖ψ‖ := by
  set s := φ + ψ with hs
  have hAφ : A φ = ψ - A ψ - φ := by rw [← hrel]; abel
  have hd : φ - ψ = -A s := by rw [hs, map_add, hAφ]; abel
  have e1 : RCLike.re ⟪φ - ψ, s⟫_ℂ = ‖φ‖ ^ 2 - ‖ψ‖ ^ 2 := by
    have hcs : RCLike.re ⟪ψ, φ⟫_ℂ = RCLike.re ⟪φ, ψ⟫_ℂ := inner_re_symm ψ φ
    rw [hs, inner_sub_left, inner_add_right, inner_add_right, map_sub, map_add, map_add,
      inner_self_eq_norm_sq, inner_self_eq_norm_sq, hcs]
    ring
  have e2 : RCLike.re ⟪φ - ψ, s⟫_ℂ = 0 := by
    have hs0 : RCLike.re ⟪A s, s⟫_ℂ = 0 := by rw [hs]; exact hlocal
    rw [hd, inner_neg_left, map_neg, hs0, neg_zero]
  have key : ‖φ‖ ^ 2 = ‖ψ‖ ^ 2 := by rw [e2] at e1; linarith
  rw [← Real.sqrt_sq (norm_nonneg φ), ← Real.sqrt_sq (norm_nonneg ψ), key]

/-- **The Cayley step preserves the length.**

    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]  [a complex inner-product space]
    (A : E →ₗ[ℂ] E)          [a complex-linear map of the space to itself]
    (hskew : ∀ x y, ⟪A x, y⟫_ℂ = -⟪x, A y⟫_ℂ)
                             [the map is skew-adjoint]
    ∀                        [for every single one]
    (φ ψ : E)                [the point before the step and the point after it]
    (hrel : φ + A φ = ψ - A ψ)
                             [the Cayley relation: the step solves this for the two points]
    ‖φ‖ = ‖ψ‖                [the two points have the same length]

    So: `skew_re_zero` at `φ + ψ` discharges the local real-part premise of
    `cayley_isometry_of_re_zero`. This is the predecessor's `cayley_isometry`, the discrete
    unitarity of the Cayley step, stated with neither a finite dimension nor a total inverse of
    `1 + A`. -/
theorem cayley_isometry (A : E →ₗ[ℂ] E)
    (hskew : ∀ x y, ⟪A x, y⟫_ℂ = -⟪x, A y⟫_ℂ) (φ ψ : E)
    (hrel : φ + A φ = ψ - A ψ) :
    ‖φ‖ = ‖ψ‖ :=
  cayley_isometry_of_re_zero A φ ψ hrel (skew_re_zero A hskew (φ + ψ))

end Trichotomy
end Apeiron
