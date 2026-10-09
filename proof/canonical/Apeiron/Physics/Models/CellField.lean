import Apeiron.Trichotomy.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# The cone, the ray, and the shell

@name CellField
@cite euclid-elements-ii-5
@cite kisil-2010
@grade proved

A field carries a signal from place to place, and what fixes how fast is the set of points of
size zero. `Trichotomy.exists_normForm_zero_iff_disc_nonneg` says that set holds something other
than the origin exactly when `σ` is at zero or above, so a signal that travels at all needs
the non-negative face. On the elliptic face the only null point is the origin and there is
nothing to travel along.

Above zero the null set is two lines, `z.1 = ±√σ · z.2` (`cone_rays`), and those are the field's
characteristics. A profile carried rigidly along one of them, `f(x − √σ·t)`, is constant on the
ray (`mover_constant_on_ray`) and solves the first-order transport law
`∂_t ψ + √σ·∂_x ψ = 0` (`mover_transport`). Nothing outruns the ray, because the value at a place
is the value that entered the ray.

A wave that is not carried rigidly is detuned off the ray, and the amount is the norm form. For
`ψ = cos(kx − ωt)` the second-order residual factors in one line,

```
   ∂_tt ψ − σ·∂_xx ψ + m²·ψ  =  (m² − normForm σ (ω, k)) · ψ
```

(`kg_residual`), so the field law holds exactly on the shell `normForm σ (ω, k) = m²`, at any
place where the wave is not passing through zero (`planeWave_law_iff_on_shell`). The shell is a
level set of the algebra's own norm form, and the massless shell is the cone.

The non-vanishing hypothesis is load-bearing and `law_holds_off_shell_at_a_node` is the witness:
at a zero of the cosine the equation holds whatever the frequency and momentum are, and says
nothing.

## Root

Instantiates `Trichotomy.normForm` from `Apeiron/Trichotomy/Basic.lean`, and rides
`Trichotomy.exists_normForm_zero_iff_disc_nonneg` for which faces have a cone at all and
`Trichotomy.hasNullPair_iff_disc_pos` for which have two independent rays. The edges:

* `cone_rays` is the factorisation `a² − σb² = (a − √σ b)(a + √σ b)`, which is Euclid II.5 at
  `σ = 1` and is stated here at every non-negative setting.
* `massless_needs_disc_nonneg` is `exists_normForm_zero_iff_disc_nonneg` read as a statement
  about a massless excitation: below zero there is none with nonzero momentum.
* `kg_residual` puts `Trichotomy.normForm σ (ω, k)` in the field law, so the mass shell is the
  norm form rather than resembling it.

The statements about a ray and about the plane wave are on the non-negative faces, since `√σ` is
what they are carried by; `kg_residual` and `planeWave_law_iff_on_shell` hold at every setting.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- **At or above zero the points of size zero are two rays.**

    σ                                   [the parameter]
    ℝ                                   [the real numbers: every number on the number line]
    ≤                                   [is smaller than or equal to]
    ↔                                   [exactly when, true in both directions]
    ∨                                   [at least one is true]
    normForm                            [the norm form: a point times its conjugate]
    Cell                                [a point: two real numbers written side by side]
    Real.sqrt                           [the number that squares to give this]

    So: the cone is the pair of lines the difference of two squares factors into. Euclid's
    `a² − b² = (a+b)(a−b)` is this at `σ = 1`, and the general statement needs only that `σ`
    has a square root. -/
theorem cone_rays {σ : ℝ} (hσ : 0 ≤ σ) (z : Cell) :
    normForm σ z = 0 ↔ z.1 = Real.sqrt σ * z.2 ∨ z.1 = -(Real.sqrt σ * z.2) := by
  have hsq : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt hσ
  constructor
  · intro h
    have hfac : (z.1 - Real.sqrt σ * z.2) * (z.1 + Real.sqrt σ * z.2) = 0 := by
      simp only [normForm] at h
      linear_combination h - z.2 ^ 2 * hsq
    rcases mul_eq_zero.mp hfac with hc | hc
    · exact Or.inl (by linarith)
    · exact Or.inr (by linarith)
  · rintro (h | h) <;> simp only [normForm, AnyRing.ringNormForm] <;> rw [h] <;>
      linear_combination (z.2 ^ 2) * hsq

/-- **A massless excitation with momentum needs `σ ≥ 0`.**

    σ                                   [the parameter]
    ℝ                                   [the real numbers: every number on the number line]
    ∃                                   [there is at least one]
    ≠                                   [is not equal to]
    ∧                                   [both are true]
    ↔                                   [exactly when, true in both directions]
    ≤                                   [is smaller than or equal to]
    normForm                            [the norm form: a point times its conjugate]
    Cell                                [a point: two real numbers written side by side]

    So: `exists_normForm_zero_iff_disc_nonneg` read on the shell. Below zero the only point of
    size zero is the origin, so a massless excitation has no frequency and no momentum and is
    not there at all. The cone is what a signal travels on and the elliptic face has none. -/
theorem massless_needs_disc_nonneg (σ : ℝ) :
    (∃ z : Cell, z ≠ 0 ∧ normForm σ z = 0) ↔ 0 ≤ σ :=
  exists_normForm_zero_iff_disc_nonneg σ

/-- A profile carried rigidly along a ray.

    σ                                   [the parameter]
    ℝ                                   [the real numbers: every number on the number line]
    →                                   [takes the left and gives the right]
    f (x - Real.sqrt σ * t)             [the profile, read at the place the ray came from]
    mover                               [a profile carried rigidly along a ray]
    Real.sqrt                           [the number that squares to give this] -/
noncomputable def mover (σ : ℝ) (f : ℝ → ℝ) (t x : ℝ) : ℝ := f (x - Real.sqrt σ * t)

/-- **The profile is the same everywhere along one ray.**

    σ                                                       [the parameter]
    ℝ                                                       [the real numbers]
    →                                                       [takes the left and gives the right]
    x - Real.sqrt σ * t = x' - Real.sqrt σ * t'              [the two places lie on one ray]
    mover                                                   [a profile carried along a ray]
    Real.sqrt                                               [the number that squares to this]

    So: the value at a place is the value that entered the ray, so nothing arrives faster than
    the ray does. -/
theorem mover_constant_on_ray (σ : ℝ) (f : ℝ → ℝ) (t x t' x' : ℝ)
    (h : x - Real.sqrt σ * t = x' - Real.sqrt σ * t') :
    mover σ f t x = mover σ f t' x' := by
  unfold mover; rw [h]

/-- **The rigid profile solves the first-order transport law.**

    σ                                                            [the parameter]
    ℝ                                                            [the real numbers]
    →                                                            [takes the left and gives right]
    ∀                                                            [for every single one]
    HasDerivAt                                                   [is changing at that rate]
    deriv                                                        [how fast the thing is changing]
    mover                                                        [a profile carried along a ray]
    Real.sqrt                                                    [the number that squares to this]

    So: the time rate and the place rate are equal and opposite up to the ray's speed, which is
    what "carried rigidly along the ray" says as an equation. -/
theorem mover_transport (σ : ℝ) (f df : ℝ → ℝ) (t x : ℝ)
    (hf : ∀ z, HasDerivAt f (df z) z) :
    deriv (fun s => mover σ f s x) t + Real.sqrt σ * deriv (fun u => mover σ f t u) x = 0 := by
  have hin_t : HasDerivAt (fun s : ℝ => x - Real.sqrt σ * s) (-Real.sqrt σ) t := by
    simpa using (((hasDerivAt_id t).const_mul (Real.sqrt σ)).const_sub x)
  have hin_x : HasDerivAt (fun u : ℝ => u - Real.sqrt σ * t) (1 : ℝ) x :=
    (hasDerivAt_id x).sub_const _
  have ht : HasDerivAt (fun s => mover σ f s x)
      (df (x - Real.sqrt σ * t) * -Real.sqrt σ) t :=
    (hf (x - Real.sqrt σ * t)).comp t hin_t
  have hx : HasDerivAt (fun u => mover σ f t u) (df (x - Real.sqrt σ * t) * 1) x :=
    (hf (x - Real.sqrt σ * t)).comp x hin_x
  rw [ht.deriv, hx.deriv]
  ring

/-- A single travelling mode.

    ℝ                                       [the real numbers: every number on the number line]
    Real.cos (k * x - ω * t)                [the mode at that time and place]
    planeWave                               [a single travelling mode] -/
noncomputable def planeWave (ω k t x : ℝ) : ℝ := Real.cos (k * x - ω * t)

/-- The mode's second rate of change in time is minus the frequency squared times itself.

    ℝ                                                   [the real numbers]
    deriv                                               [how fast the thing is changing]
    planeWave                                           [a single travelling mode] -/
theorem planeWave_dtt (ω k t x : ℝ) :
    deriv (fun s => deriv (fun r => planeWave ω k r x) s) t = -(ω ^ 2) * planeWave ω k t x := by
  have hin : ∀ s : ℝ, HasDerivAt (fun r : ℝ => k * x - ω * r) (-ω) s := fun s => by
    simpa using ((hasDerivAt_id s).const_mul ω).const_sub (k * x)
  have h1 : ∀ s : ℝ, HasDerivAt (fun r => planeWave ω k r x)
      (-Real.sin (k * x - ω * s) * -ω) s := fun s =>
    (Real.hasDerivAt_cos _).comp s (hin s)
  have hd : (fun s => deriv (fun r => planeWave ω k r x) s)
      = fun s => -Real.sin (k * x - ω * s) * -ω :=
    funext fun s => (h1 s).deriv
  rw [hd]
  have h2 : HasDerivAt (fun s : ℝ => -Real.sin (k * x - ω * s) * -ω)
      (-(Real.cos (k * x - ω * t) * -ω) * -ω) t :=
    (((Real.hasDerivAt_sin _).comp t (hin t)).neg).mul_const (-ω)
  rw [h2.deriv]
  simp only [planeWave]
  ring

/-- The mode's second rate of change in place is minus the momentum squared times itself.

    ℝ                                                   [the real numbers]
    deriv                                               [how fast the thing is changing]
    planeWave                                           [a single travelling mode] -/
theorem planeWave_dxx (ω k t x : ℝ) :
    deriv (fun u => deriv (fun v => planeWave ω k t v) u) x = -(k ^ 2) * planeWave ω k t x := by
  have hin : ∀ u : ℝ, HasDerivAt (fun v : ℝ => k * v - ω * t) k u := fun u => by
    simpa using ((hasDerivAt_id u).const_mul k).sub_const (ω * t)
  have h1 : ∀ u : ℝ, HasDerivAt (fun v => planeWave ω k t v)
      (-Real.sin (k * u - ω * t) * k) u := fun u =>
    (Real.hasDerivAt_cos _).comp u (hin u)
  have hd : (fun u => deriv (fun v => planeWave ω k t v) u)
      = fun u => -Real.sin (k * u - ω * t) * k :=
    funext fun u => (h1 u).deriv
  rw [hd]
  have h2 : HasDerivAt (fun u : ℝ => -Real.sin (k * u - ω * t) * k)
      (-(Real.cos (k * x - ω * t) * k) * k) x :=
    (((Real.hasDerivAt_sin _).comp x (hin x)).neg).mul_const k
  rw [h2.deriv]
  simp only [planeWave]
  ring

/-- **The field law's residual is the distance from the shell, times the mode.**

    σ                                                            [the parameter]
    ℝ                                                            [the real numbers]
    deriv                                                        [how fast the thing is changing]
    normForm                                                     [the norm form of a point]
    planeWave                                                    [a single travelling mode]

    So: one identity, and the whole file reads off it. The quantity the law measures is the
    algebra's own norm form of the frequency and momentum pair, against the mass squared. -/
theorem kg_residual (σ ω k m t x : ℝ) :
    deriv (fun s => deriv (fun r => planeWave ω k r x) s) t
        - σ * deriv (fun u => deriv (fun v => planeWave ω k t v) u) x
        + m ^ 2 * planeWave ω k t x
      = (m ^ 2 - normForm σ (ω, k)) * planeWave ω k t x := by
  rw [planeWave_dtt, planeWave_dxx]
  simp only [normForm]
  ring

/-- **Off a node, the mode solves the field law exactly on the shell.**

    σ                                                            [the parameter]
    ℝ                                                            [the real numbers]
    ≠                                                            [is not equal to]
    ↔                                                            [exactly when, both directions]
    deriv                                                        [how fast the thing is changing]
    normForm                                                     [the norm form of a point]
    planeWave                                                    [a single travelling mode]

    So: the mass shell is a level set of the norm form, at every value of `σ`. At `σ = 1`
    it is the relativistic shell, at `σ = 0` the non-dispersive one, and below zero the shell has
    no massless member because `massless_needs_disc_nonneg` says the cone is empty there. -/
theorem planeWave_law_iff_on_shell (σ ω k m t x : ℝ) (hψ : planeWave ω k t x ≠ 0) :
    (deriv (fun s => deriv (fun r => planeWave ω k r x) s) t
        - σ * deriv (fun u => deriv (fun v => planeWave ω k t v) u) x
        + m ^ 2 * planeWave ω k t x = 0)
      ↔ normForm σ (ω, k) = m ^ 2 := by
  rw [kg_residual]
  constructor
  · intro h
    rcases mul_eq_zero.mp h with hc | hc
    · linarith
    · exact absurd hc hψ
  · intro h
    rw [h]
    ring

/-- **The non-vanishing hypothesis carries weight: at a node the law says nothing.**

    ℝ                                                            [the real numbers]
    ∧                                                            [both are true]
    ≠                                                            [is not equal to]
    deriv                                                        [how fast the thing is changing]
    normForm                                                     [the norm form of a point]
    planeWave                                                    [a single travelling mode]

    So: at a zero of the mode the residual vanishes whatever the frequency and momentum are, so
    the converse without the hypothesis is false. The witness is the mode at rest, read a quarter
    turn along. -/
theorem law_holds_off_shell_at_a_node :
    planeWave 0 1 0 (Real.pi / 2) = 0
      ∧ (deriv (fun s => deriv (fun r => planeWave 0 1 r (Real.pi / 2)) s) 0
          - 1 * deriv (fun u => deriv (fun v => planeWave 0 1 0 v) u) (Real.pi / 2)
          + 0 ^ 2 * planeWave 0 1 0 (Real.pi / 2) = 0)
      ∧ normForm 1 ((0 : ℝ), (1 : ℝ)) ≠ 0 ^ 2 := by
  have hzero : planeWave 0 1 0 (Real.pi / 2) = 0 := by
    simp [planeWave]
  refine ⟨hzero, ?_, ?_⟩
  · rw [kg_residual, hzero]
    ring
  · simp [normForm]

end Physics
end Apeiron
