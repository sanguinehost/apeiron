/-
# Face contraction certificates for the affine state recurrence over `ℝ`

The recurrent state update of the model ladder is the affine step
`h_t = ringMul σ a_t h_{t-1} + b_t` in `RingCell ℝ = ℝ × ℝ`
(`Apeiron.Trichotomy.AnyRing.ringMul`). This module supplies, on each of the three real faces of
`R[x]/(x² − σ)`, a metric `d_σ` and a rate `ρ_σ` with

    d_σ (ringMul σ a x) (ringMul σ a y) ≤ ρ_σ a * d_σ x y,

and then the two orbit consequences for a list of steps: the homogeneous difference bound and the
additively driven bound. The `b_t` cancel in the difference of two trajectories, so the rates
control the state update.

## Faces

* **Elliptic** `σ = -1`: `d` is the Euclidean magnitude `√(x₀² + x₁²)`, `ρ a = √(a₀² + a₁²)`.
  The norm form `a₀² − σ a₁² = a₀² + a₁²` is the squared rate, and it is multiplicative:
  `euclidMagSq_ringMul_neg_one` is `‖ringMul (-1) a x‖² = (a₀² + a₁²) ‖x‖²`.
* **Hyperbolic** `σ = 1`: `d` is `max(|x₀ + x₁|, |x₀ − x₁|)`, `ρ a = max(|a₀ + a₁|, |a₀ − a₁|)`.
  The two channel readings (`ringUpper 1`, `ringLower 1`) multiply independently
  (`ringUpper_mul`, `ringLower_mul`).
* **Parabolic** `σ = 0`: for a weight `c > 0` the metric is `d_c(x) = |x₀| c + |x₁|` and the rate
  is `ρ_c a = |a₀| + |a₁| / c`. The bound holds for every `c > 0`, and `|a₀| < 1` admits a `c` with
  `ρ_c a < 1` (`paraRate_exists_lt_one`).

## Grade and evidence

Every statement below carries grade `proved`: it is a Lean kernel check with no `sorry`, `admit`,
`axiom`, `opaque` or `set_option`, and each headline's axiom cone is
`{propext, Classical.choice, Quot.sound}` under `#print axioms`. The metric and rate are the
certificate objects the product ladder R3 requires of the state update.

## Trust boundary

The statements are about the real-valued reference `ℝ`. They do not reach `rustc`/LLVM, the kernel
ABI, the fixed-point extraction (`Cell<i64>`) or any GPU target: contraction of the real recurrence
is a counting statement about the real trajectory, not a measurement.
-/
import Nous.Model.Decomposition
import Mathlib.Algebra.Field.GeomSum
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open scoped BigOperators

noncomputable section

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing

/-! ## The affine step, its folds, and the shared linear identity -/

/-- One affine step `h ↦ ringMul σ a h + b` of the recurrence, carried as a pair `(a, b)`.

    σ        [the quadratic parameter, an element of `ℝ`]
    h        [the current state]
    (a, b)   [the step coefficient and the additive drive] -/
def runStep (σ : ℝ) (h : RingCell ℝ) (p : RingCell ℝ × RingCell ℝ) : RingCell ℝ :=
  ringMul σ p.1 h + p.2

/-- The state after folding a list of steps from the left. Temporal cost is one `runStep` per step;
    the list is consumed once. -/
def trajectory (σ : ℝ) (h₀ : RingCell ℝ) (steps : List (RingCell ℝ × RingCell ℝ)) :
    RingCell ℝ :=
  steps.foldl (runStep σ) h₀

/-- The homogeneous part of the fold: the same steps with the additive drive dropped. -/
def homRun (σ : ℝ) (e : RingCell ℝ) (steps : List (RingCell ℝ × RingCell ℝ)) : RingCell ℝ :=
  steps.foldl (fun e p => ringMul σ p.1 e) e

theorem trajectory_cons (σ : ℝ) (h : RingCell ℝ) (p : RingCell ℝ × RingCell ℝ)
    (rest : List (RingCell ℝ × RingCell ℝ)) :
    trajectory σ h (p :: rest) = trajectory σ (runStep σ h p) rest := rfl

theorem homRun_cons (σ : ℝ) (e : RingCell ℝ) (p : RingCell ℝ × RingCell ℝ)
    (rest : List (RingCell ℝ × RingCell ℝ)) :
    homRun σ e (p :: rest) = homRun σ (ringMul σ p.1 e) rest := rfl

/-- `ringMul σ a` is additive in its point argument: the difference of two products is the product
    of the difference. -/
theorem ringMul_sub (σ : ℝ) (a x y : RingCell ℝ) :
    ringMul σ a (x - y) = ringMul σ a x - ringMul σ a y := by
  apply Prod.ext <;>
    simp only [ringMul, Prod.fst_sub, Prod.snd_sub] <;>
    ring

/-- The affine drives cancel: two trajectories over the same steps differ by the homogeneous fold
    of their initial difference. -/
theorem trajectory_sub (σ : ℝ) (h₀ h₀' : RingCell ℝ)
    (steps : List (RingCell ℝ × RingCell ℝ)) :
    trajectory σ h₀ steps - trajectory σ h₀' steps = homRun σ (h₀ - h₀') steps := by
  induction steps generalizing h₀ h₀' with
  | nil => rfl
  | cons p rest ih =>
      rw [trajectory_cons, trajectory_cons, ih, homRun_cons]
      congr 1
      apply Prod.ext <;>
        simp only [runStep, ringMul, Prod.fst_add, Prod.snd_add, Prod.fst_sub, Prod.snd_sub] <;>
        ring

/-! ## Elliptic face `σ = -1` -/

/-- Squared Euclidean magnitude `x₀² + x₁²`. -/
def euclidMagSq (x : RingCell ℝ) : ℝ := x.1 ^ 2 + x.2 ^ 2

/-- Euclidean magnitude `√(x₀² + x₁²)`. -/
def euclidMag (x : RingCell ℝ) : ℝ := Real.sqrt (euclidMagSq x)

/-- Euclidean distance between two ring points. -/
def euclidDist (x y : RingCell ℝ) : ℝ := euclidMag (x - y)

/-- Elliptic contraction rate `√(a₀² + a₁²)`. -/
def euclidRate (a : RingCell ℝ) : ℝ := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

theorem euclidMag_nonneg (x : RingCell ℝ) : 0 ≤ euclidMag x := Real.sqrt_nonneg _

theorem euclidMag_sq (x : RingCell ℝ) : euclidMag x ^ 2 = euclidMagSq x :=
  Real.sq_sqrt (by unfold euclidMagSq; positivity)

/-- Multiplicativity of the elliptic norm form: the squared Euclidean magnitude of
    `ringMul (-1) a x` is `(a₀² + a₁²)` times that of `x`. -/
theorem euclidMagSq_ringMul_neg_one (a x : RingCell ℝ) :
    euclidMagSq (ringMul (-1) a x) = (a.1 ^ 2 + a.2 ^ 2) * euclidMagSq x := by
  simp only [euclidMagSq]
  ring

/-- The Euclidean magnitude scales by `√(a₀² + a₁²)` under `ringMul (-1) a`. -/
theorem euclidMag_ringMul_neg_one (a x : RingCell ℝ) :
    euclidMag (ringMul (-1) a x) = euclidRate a * euclidMag x := by
  simp only [euclidMag, euclidRate]
  rw [euclidMagSq_ringMul_neg_one,
    Real.sqrt_mul (by positivity : (0 : ℝ) ≤ a.1 ^ 2 + a.2 ^ 2)]

/-- Elliptic contraction certificate: `ringMul (-1) a` contracts the Euclidean distance by
    `euclidRate a`. -/
theorem euclid_contraction (a x y : RingCell ℝ) :
    euclidDist (ringMul (-1) a x) (ringMul (-1) a y) ≤ euclidRate a * euclidDist x y := by
  rw [euclidDist, euclidDist, ← ringMul_sub (-1) a x y, euclidMag_ringMul_neg_one]

/-- The Euclidean magnitude is subadditive: `‖x + y‖ ≤ ‖x‖ + ‖y‖`. Temporal and spatial cost are
    `O(1)`. -/
theorem euclidMag_add_le (x y : RingCell ℝ) :
    euclidMag (x + y) ≤ euclidMag x + euclidMag y := by
  have hcs : (x.1 * y.1 + x.2 * y.2) ^ 2 ≤ euclidMagSq x * euclidMagSq y := by
    simp only [euclidMagSq]
    nlinarith [sq_nonneg (x.1 * y.2 - x.2 * y.1)]
  have hr : x.1 * y.1 + x.2 * y.2 ≤ euclidMag x * euclidMag y := by
    have h1 : x.1 * y.1 + x.2 * y.2 ≤ Real.sqrt (euclidMagSq x * euclidMagSq y) :=
      Real.le_sqrt_of_sq_le hcs
    rw [Real.sqrt_mul (by unfold euclidMagSq; positivity) (euclidMagSq y)] at h1
    simpa only [euclidMag] using h1
  have hxy : euclidMagSq (x + y) ≤ (euclidMag x + euclidMag y) ^ 2 := by
    have hexp : euclidMagSq (x + y)
        = euclidMagSq x + 2 * (x.1 * y.1 + x.2 * y.2) + euclidMagSq y := by
      simp only [euclidMagSq, Prod.fst_add, Prod.snd_add]
      ring
    rw [hexp]
    nlinarith [hr, euclidMag_sq x, euclidMag_sq y]
  calc euclidMag (x + y) = Real.sqrt (euclidMagSq (x + y)) := rfl
    _ ≤ Real.sqrt ((euclidMag x + euclidMag y) ^ 2) := Real.sqrt_le_sqrt hxy
    _ = euclidMag x + euclidMag y := Real.sqrt_sq (add_nonneg (euclidMag_nonneg x) (euclidMag_nonneg y))

/-! ## Hyperbolic face `σ = 1` -/

/-- Hyperbolic magnitude: the larger of the two channel readings `|x₀ + x₁|`, `|x₀ − x₁|`. -/
def hyperMag (x : RingCell ℝ) : ℝ := max |ringUpper 1 x| |ringLower 1 x|

/-- Hyperbolic distance between two ring points. -/
def hyperDist (x y : RingCell ℝ) : ℝ := hyperMag (x - y)

/-- Hyperbolic contraction rate `max(|a₀ + a₁|, |a₀ − a₁|)`. -/
def hyperRate (a : RingCell ℝ) : ℝ := max |ringUpper 1 a| |ringLower 1 a|

/-- The channel readings at `r = 1` are the two coordinate combinations. -/
theorem ringUpper_one (z : RingCell ℝ) : ringUpper 1 z = z.1 + z.2 := by simp [ringUpper]

theorem ringLower_one (z : RingCell ℝ) : ringLower 1 z = z.1 - z.2 := by simp [ringLower]

theorem hyperMag_nonneg (x : RingCell ℝ) : 0 ≤ hyperMag x :=
  le_trans (abs_nonneg _) (le_max_left _ _)

/-- The hyperbolic magnitude scales by `hyperRate a`: the channel readings multiply independently
    (`ringUpper_mul`, `ringLower_mul`). -/
theorem hyperMag_ringMul_one (a z : RingCell ℝ) :
    hyperMag (ringMul 1 a z) ≤ hyperRate a * hyperMag z := by
  have hu : ringUpper 1 (ringMul 1 a z) = ringUpper 1 a * ringUpper 1 z :=
    ringUpper_mul (σ := 1) (r := 1) (by norm_num) a z
  have hl : ringLower 1 (ringMul 1 a z) = ringLower 1 a * ringLower 1 z :=
    ringLower_mul (σ := 1) (r := 1) (by norm_num) a z
  unfold hyperMag hyperRate
  rw [hu, hl, abs_mul, abs_mul]
  apply max_le
  · exact mul_le_mul (le_max_left _ _) (le_max_left _ _) (abs_nonneg _)
      (le_trans (abs_nonneg _) (le_max_left _ _))
  · exact mul_le_mul (le_max_right _ _) (le_max_right _ _) (abs_nonneg _)
      (le_trans (abs_nonneg _) (le_max_right _ _))

/-- Hyperbolic contraction certificate. -/
theorem hyper_contraction (a x y : RingCell ℝ) :
    hyperDist (ringMul 1 a x) (ringMul 1 a y) ≤ hyperRate a * hyperDist x y := by
  rw [hyperDist, hyperDist, ← ringMul_sub 1 a x y]
  exact hyperMag_ringMul_one a (x - y)

/-- The hyperbolic magnitude is subadditive. -/
theorem hyperMag_add_le (x y : RingCell ℝ) :
    hyperMag (x + y) ≤ hyperMag x + hyperMag y := by
  have hu : ringUpper 1 (x + y) = ringUpper 1 x + ringUpper 1 y := ringUpper_add 1 x y
  have hl : ringLower 1 (x + y) = ringLower 1 x + ringLower 1 y := ringLower_add 1 x y
  unfold hyperMag
  rw [hu, hl]
  apply max_le
  · calc |ringUpper 1 x + ringUpper 1 y| ≤ |ringUpper 1 x| + |ringUpper 1 y| := abs_add_le _ _
      _ ≤ max |ringUpper 1 x| |ringLower 1 x| + max |ringUpper 1 y| |ringLower 1 y| :=
            add_le_add (le_max_left _ _) (le_max_left _ _)
  · calc |ringLower 1 x + ringLower 1 y| ≤ |ringLower 1 x| + |ringLower 1 y| := abs_add_le _ _
      _ ≤ max |ringUpper 1 x| |ringLower 1 x| + max |ringUpper 1 y| |ringLower 1 y| :=
            add_le_add (le_max_right _ _) (le_max_right _ _)

/-! ## Parabolic face `σ = 0` -/

/-- Weighted parabolic magnitude `|x₀| c + |x₁|` at weight `c`. -/
def paraMag (c : ℝ) (x : RingCell ℝ) : ℝ := |x.1| * c + |x.2|

/-- Weighted parabolic distance between two ring points. -/
def paraDist (c : ℝ) (x y : RingCell ℝ) : ℝ := paraMag c (x - y)

/-- Parabolic contraction rate `|a₀| + |a₁| / c` at weight `c`. -/
def paraRate (c : ℝ) (a : RingCell ℝ) : ℝ := |a.1| + |a.2| / c

theorem paraMag_nonneg (c : ℝ) (hc : 0 ≤ c) (x : RingCell ℝ) : 0 ≤ paraMag c x := by
  unfold paraMag
  positivity

/-- The Jordan-block product `ringMul 0 a x = (a₀ x₀, a₀ x₁ + a₁ x₀)` contracts the weighted
    parabolic magnitude at rate `|a₀| + |a₁| / c`, for every weight `c > 0`. -/
theorem paraMag_ringMul_zero (c : ℝ) (hc : 0 < c) (a x : RingCell ℝ) :
    paraMag c (ringMul 0 a x) ≤ paraRate c a * paraMag c x := by
  have h12 : |a.1 * x.2 + a.2 * x.1| ≤ |a.1| * |x.2| + |a.2| * |x.1| := by
    calc |a.1 * x.2 + a.2 * x.1| ≤ |a.1 * x.2| + |a.2 * x.1| := abs_add_le _ _
      _ = |a.1| * |x.2| + |a.2| * |x.1| := by rw [abs_mul, abs_mul]
  have hR : (|a.1| + |a.2| / c) * (|x.1| * c + |x.2|)
      = |a.1| * |x.1| * c + |a.1| * |x.2| + |a.2| * |x.1| + (|a.2| / c) * |x.2| := by
    field_simp
    ring
  simp only [paraMag, paraRate, zero_mul, add_zero]
  rw [abs_mul]
  calc |a.1| * |x.1| * c + |a.1 * x.2 + a.2 * x.1|
      ≤ |a.1| * |x.1| * c + (|a.1| * |x.2| + |a.2| * |x.1|) :=
          add_le_add (le_refl _) h12
    _ ≤ (|a.1| + |a.2| / c) * (|x.1| * c + |x.2|) := by
          rw [hR]
          have : 0 ≤ (|a.2| / c) * |x.2| := by positivity
          linarith

/-- Parabolic contraction certificate down a pair of points. -/
theorem para_contraction (c : ℝ) (hc : 0 < c) (a x y : RingCell ℝ) :
    paraDist c (ringMul 0 a x) (ringMul 0 a y) ≤ paraRate c a * paraDist c x y := by
  rw [paraDist, paraDist, ← ringMul_sub 0 a x y]
  exact paraMag_ringMul_zero c hc a (x - y)

/-- The weighted parabolic magnitude is subadditive for `c ≥ 0`. -/
theorem paraMag_add_le (c : ℝ) (hc : 0 ≤ c) (x y : RingCell ℝ) :
    paraMag c (x + y) ≤ paraMag c x + paraMag c y := by
  have h0 : |(x + y).1| * c ≤ |x.1| * c + |y.1| * c := by
    calc |(x + y).1| * c ≤ (|x.1| + |y.1|) * c :=
          mul_le_mul_of_nonneg_right (by rw [Prod.fst_add]; exact abs_add_le _ _) hc
      _ = |x.1| * c + |y.1| * c := by ring
  have h1 : |(x + y).2| ≤ |x.2| + |y.2| := by rw [Prod.snd_add]; exact abs_add_le _ _
  simp only [paraMag]
  linarith

/-- When `|a₀| < 1`, some weight `c > 0` makes the parabolic rate strictly below one. -/
theorem paraRate_exists_lt_one (a : RingCell ℝ) (ha : |a.1| < 1) :
    ∃ c : ℝ, 0 < c ∧ paraRate c a < 1 := by
  set δ : ℝ := (1 - |a.1|) / 2 with hδ
  have hδ0 : 0 < δ := by rw [hδ]; linarith
  refine ⟨|a.2| / δ + 1, by positivity, ?_⟩
  have hle : |a.2| / (|a.2| / δ + 1) ≤ δ := by
    rw [div_le_iff₀ (by positivity : (0 : ℝ) < |a.2| / δ + 1)]
    have hmul : δ * (|a.2| / δ + 1) = |a.2| + δ := by field_simp
    rw [hmul]
    linarith [abs_nonneg a.2]
  unfold paraRate
  linarith [hδ, hle, ha]

/-! ## Orbit bounds for any face -/

/-- The homogeneous fold is bounded by `ρ ^ n` times the initial magnitude, when every step's rate
    is at most `ρ`. -/
theorem homRun_bound (σ : ℝ) (w rate : RingCell ℝ → ℝ) (ρ : ℝ)
    (hw_nonneg : ∀ z, 0 ≤ w z)
    (hstep : ∀ a z, w (ringMul σ a z) ≤ rate a * w z)
    (hρ0 : 0 ≤ ρ)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, rate p.1 ≤ ρ)
    (e : RingCell ℝ) :
    w (homRun σ e steps) ≤ ρ ^ steps.length * w e := by
  induction steps generalizing e with
  | nil => simp [homRun]
  | cons p rest ih =>
      rw [homRun_cons]
      calc w (homRun σ (ringMul σ p.1 e) rest)
          ≤ ρ ^ rest.length * w (ringMul σ p.1 e) :=
            ih (fun q hq => hsteps q (List.mem_cons_of_mem p hq)) (ringMul σ p.1 e)
        _ ≤ ρ ^ rest.length * (rate p.1 * w e) :=
            mul_le_mul_of_nonneg_left (hstep p.1 e) (pow_nonneg hρ0 _)
        _ ≤ ρ ^ rest.length * (ρ * w e) :=
            mul_le_mul_of_nonneg_left
              (mul_le_mul_of_nonneg_right (hsteps p (by simp)) (hw_nonneg e))
              (pow_nonneg hρ0 _)
        _ = ρ ^ (p :: rest).length * w e := by
            rw [List.length_cons, pow_succ]
            ring

/-- Contraction of two trajectories over the same steps: the affine drives cancel and the
    accumulated rate bounds the initial distance. -/
theorem contraction_orbit (σ : ℝ) (d : RingCell ℝ → RingCell ℝ → ℝ) (w rate : RingCell ℝ → ℝ)
    (ρ : ℝ)
    (hd : ∀ x y, d x y = w (x - y))
    (hw_nonneg : ∀ z, 0 ≤ w z)
    (hstep : ∀ a z, w (ringMul σ a z) ≤ rate a * w z)
    (hρ0 : 0 ≤ ρ)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, rate p.1 ≤ ρ)
    (h₀ h₀' : RingCell ℝ) :
    d (trajectory σ h₀ steps) (trajectory σ h₀' steps) ≤ ρ ^ steps.length * d h₀ h₀' := by
  rw [hd, hd, trajectory_sub]
  exact homRun_bound σ w rate ρ hw_nonneg hstep hρ0 steps hsteps (h₀ - h₀')

/-- Additively driven orbit bound, sharp finite form: the accumulated drive sum
    `B (1 + ρ + … + ρ^{n-1})`. -/
theorem bounded_orbit_sharp (σ : ℝ) (w rate : RingCell ℝ → ℝ) (ρ B : ℝ)
    (hw_nonneg : ∀ z, 0 ≤ w z)
    (hw_add : ∀ x y, w (x + y) ≤ w x + w y)
    (hstep : ∀ a z, w (ringMul σ a z) ≤ rate a * w z)
    (hρ0 : 0 ≤ ρ)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, rate p.1 ≤ ρ)
    (hb : ∀ p ∈ steps, w p.2 ≤ B)
    (h₀ : RingCell ℝ) :
    w (trajectory σ h₀ steps)
      ≤ ρ ^ steps.length * w h₀ + B * (∑ i ∈ Finset.range steps.length, ρ ^ i) := by
  induction steps generalizing h₀ with
  | nil => simp [trajectory]
  | cons p rest ih =>
      rw [trajectory_cons]
      have hb' : ∀ q ∈ rest, w q.2 ≤ B := fun q hq => hb q (List.mem_cons_of_mem p hq)
      have hp : w p.2 ≤ B := hb p (by simp)
      have hrp : rate p.1 ≤ ρ := hsteps p (by simp)
      have hrun : w (runStep σ h₀ p) ≤ rate p.1 * w h₀ + B := by
        calc w (runStep σ h₀ p) = w (ringMul σ p.1 h₀ + p.2) := rfl
          _ ≤ w (ringMul σ p.1 h₀) + w p.2 := hw_add _ _
          _ ≤ rate p.1 * w h₀ + B := add_le_add (hstep p.1 h₀) hp
      calc w (trajectory σ (runStep σ h₀ p) rest)
          ≤ ρ ^ rest.length * w (runStep σ h₀ p)
              + B * (∑ i ∈ Finset.range rest.length, ρ ^ i) :=
            ih (fun q hq => hsteps q (List.mem_cons_of_mem p hq)) hb' (runStep σ h₀ p)
        _ ≤ ρ ^ rest.length * (rate p.1 * w h₀ + B)
              + B * (∑ i ∈ Finset.range rest.length, ρ ^ i) :=
            add_le_add (mul_le_mul_of_nonneg_left hrun (pow_nonneg hρ0 _)) (le_refl _)
        _ ≤ ρ ^ (p :: rest).length * w h₀
              + B * (∑ i ∈ Finset.range (p :: rest).length, ρ ^ i) := by
            rw [List.length_cons, geom_sum_succ', pow_succ]
            have h1 : ρ ^ rest.length * (rate p.1 * w h₀ + B)
                ≤ ρ ^ rest.length * (ρ * w h₀ + B) :=
              mul_le_mul_of_nonneg_left
                (add_le_add (mul_le_mul_of_nonneg_right hrp (hw_nonneg h₀)) le_rfl)
                (pow_nonneg hρ0 _)
            have h2 : ρ ^ rest.length * (ρ * w h₀ + B)
                = (ρ ^ rest.length * ρ) * w h₀ + B * ρ ^ rest.length := by ring
            have h3 : B * (ρ ^ rest.length + ∑ i ∈ Finset.range rest.length, ρ ^ i)
                = B * ρ ^ rest.length + B * ∑ i ∈ Finset.range rest.length, ρ ^ i := by
              ring
            linarith [h1, h2, h3]

/-- Additively driven orbit bound: with each drive bounded by `B` and each rate by `ρ < 1`, the
    magnitude grows by at most `B / (1 - ρ)`, independent of the horizon. -/
theorem bounded_orbit (σ : ℝ) (d : RingCell ℝ → RingCell ℝ → ℝ) (w rate : RingCell ℝ → ℝ)
    (ρ B : ℝ)
    (hd : ∀ x y, d x y = w (x - y))
    (hw_nonneg : ∀ z, 0 ≤ w z)
    (hw_add : ∀ x y, w (x + y) ≤ w x + w y)
    (hstep : ∀ a z, w (ringMul σ a z) ≤ rate a * w z)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hB : 0 ≤ B)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, rate p.1 ≤ ρ)
    (hb : ∀ p ∈ steps, w p.2 ≤ B)
    (h₀ : RingCell ℝ) :
    d (trajectory σ h₀ steps) 0 ≤ ρ ^ steps.length * d h₀ 0 + B / (1 - ρ) := by
  rw [hd, hd]
  simp only [sub_zero]
  have hsharp := bounded_orbit_sharp σ w rate ρ B hw_nonneg hw_add hstep hρ0 steps hsteps hb h₀
  have hden : 0 < 1 - ρ := by linarith
  have hgeom : ∑ i ∈ Finset.range steps.length, ρ ^ i ≤ 1 / (1 - ρ) := by
    rw [le_div_iff₀ hden, geom_sum_mul_neg ρ steps.length]
    have : 0 ≤ ρ ^ steps.length := pow_nonneg hρ0 _
    linarith
  have hBsum : B * (∑ i ∈ Finset.range steps.length, ρ ^ i) ≤ B * (1 / (1 - ρ)) :=
    mul_le_mul_of_nonneg_left hgeom hB
  have hdiv : B * (1 / (1 - ρ)) = B / (1 - ρ) := by simp [div_eq_mul_inv]
  linarith [hsharp, hBsum, hdiv]

/-! ## Per-face orbit corollaries -/

/-- Elliptic trajectories contract at the accumulated rate `ρ`. -/
theorem euclid_orbit (ρ : ℝ) (hρ0 : 0 ≤ ρ)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, euclidRate p.1 ≤ ρ) (h₀ h₀' : RingCell ℝ) :
    euclidDist (trajectory (-1) h₀ steps) (trajectory (-1) h₀' steps)
      ≤ ρ ^ steps.length * euclidDist h₀ h₀' :=
  contraction_orbit (-1) euclidDist euclidMag euclidRate ρ (fun _ _ => rfl) euclidMag_nonneg
    (fun a z => le_of_eq (euclidMag_ringMul_neg_one a z)) hρ0 steps hsteps h₀ h₀'

/-- Hyperbolic trajectories contract at the accumulated rate `ρ`. -/
theorem hyper_orbit (ρ : ℝ) (hρ0 : 0 ≤ ρ)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, hyperRate p.1 ≤ ρ) (h₀ h₀' : RingCell ℝ) :
    hyperDist (trajectory 1 h₀ steps) (trajectory 1 h₀' steps)
      ≤ ρ ^ steps.length * hyperDist h₀ h₀' :=
  contraction_orbit 1 hyperDist hyperMag hyperRate ρ (fun _ _ => rfl) hyperMag_nonneg
    (fun a z => hyperMag_ringMul_one a z) hρ0 steps hsteps h₀ h₀'

/-- Parabolic trajectories contract at the accumulated rate `ρ` at any weight `c > 0`. -/
theorem para_orbit (c ρ : ℝ) (hc : 0 < c) (hρ0 : 0 ≤ ρ)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, paraRate c p.1 ≤ ρ) (h₀ h₀' : RingCell ℝ) :
    paraDist c (trajectory 0 h₀ steps) (trajectory 0 h₀' steps)
      ≤ ρ ^ steps.length * paraDist c h₀ h₀' :=
  contraction_orbit 0 (paraDist c) (paraMag c) (paraRate c) ρ (fun _ _ => rfl)
    (paraMag_nonneg c hc.le) (fun a z => paraMag_ringMul_zero c hc a z) hρ0 steps hsteps h₀ h₀'

/-- Elliptic driven orbits stay within `B / (1 - ρ)` of zero. -/
theorem euclid_bounded (ρ B : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hB : 0 ≤ B)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, euclidRate p.1 ≤ ρ) (hb : ∀ p ∈ steps, euclidMag p.2 ≤ B)
    (h₀ : RingCell ℝ) :
    euclidDist (trajectory (-1) h₀ steps) 0 ≤ ρ ^ steps.length * euclidDist h₀ 0 + B / (1 - ρ) :=
  bounded_orbit (-1) euclidDist euclidMag euclidRate ρ B (fun _ _ => rfl) euclidMag_nonneg
    euclidMag_add_le (fun a z => le_of_eq (euclidMag_ringMul_neg_one a z)) hρ0 hρ1 hB steps
    hsteps hb h₀

/-- Hyperbolic driven orbits stay within `B / (1 - ρ)` of zero. -/
theorem hyper_bounded (ρ B : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hB : 0 ≤ B)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, hyperRate p.1 ≤ ρ) (hb : ∀ p ∈ steps, hyperMag p.2 ≤ B)
    (h₀ : RingCell ℝ) :
    hyperDist (trajectory 1 h₀ steps) 0 ≤ ρ ^ steps.length * hyperDist h₀ 0 + B / (1 - ρ) :=
  bounded_orbit 1 hyperDist hyperMag hyperRate ρ B (fun _ _ => rfl) hyperMag_nonneg
    hyperMag_add_le (fun a z => hyperMag_ringMul_one a z) hρ0 hρ1 hB steps hsteps hb h₀

/-- Parabolic driven orbits stay within `B / (1 - ρ)` of zero at any weight `c > 0`. -/
theorem para_bounded (c ρ B : ℝ) (hc : 0 < c) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hB : 0 ≤ B)
    (steps : List (RingCell ℝ × RingCell ℝ))
    (hsteps : ∀ p ∈ steps, paraRate c p.1 ≤ ρ) (hb : ∀ p ∈ steps, paraMag c p.2 ≤ B)
    (h₀ : RingCell ℝ) :
    paraDist c (trajectory 0 h₀ steps) 0 ≤ ρ ^ steps.length * paraDist c h₀ 0 + B / (1 - ρ) :=
  bounded_orbit 0 (paraDist c) (paraMag c) (paraRate c) ρ B (fun _ _ => rfl)
    (paraMag_nonneg c hc.le) (paraMag_add_le c hc.le) (fun a z => paraMag_ringMul_zero c hc a z)
    hρ0 hρ1 hB steps hsteps hb h₀

end Nous.Model
