import Apeiron.Trichotomy.Flow

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-!
# The face-uniform shift rule

`R[x]/(x^2 − σ)` carries one size-one flow `Apeiron.Trichotomy.flow σ`, whose two slots are

* `C σ t = (flow σ t).1`, the first slot, with `C' = σ·S`;
* `S σ t = (flow σ t).2`, the second slot, with `S' = C`;

and the second integral `K σ t = (C σ t − 1)/σ` off zero, `t²/2` at zero, chosen so that
`C σ t = 1 + σ · K σ t` holds at every sigma, zero included.

The first and second central differences of `f θ = a + b·S σ θ + c·K σ θ` are the two shift
identities `shift_first` and `shift_second`, uniform in `σ`: no face hypothesis is needed. The
three faces are instances — `σ = −1`, `s = π/2` is the parameter-shift rule, `σ = 0` is the
exact quadratic difference, `σ = 1` is the hyperbolic shift. A cubic violates `shift_first`,
which bounds the span of the identities.
-/

noncomputable section

open Real

namespace Quantum
namespace FaceShift

/-- First slot of the size-one flow: `C σ t = (flow σ t).1`. -/
abbrev C (σ t : ℝ) : ℝ := (Apeiron.Trichotomy.flow σ t).1

/-- Second slot of the size-one flow: `S σ t = (flow σ t).2`. -/
abbrev S (σ t : ℝ) : ℝ := (Apeiron.Trichotomy.flow σ t).2

/-- The second integral of the flow: `K σ t = (C σ t − 1)/σ` off zero, `t²/2` at zero.

    The zero branch is the limit of `(C σ t − 1)/σ` at `σ = 0`, and it is what makes
    `C σ t = 1 + σ · K σ t` and the two shift identities hold at `σ = 0` as well. -/
def K (σ t : ℝ) : ℝ := if σ = 0 then t ^ 2 / 2 else (C σ t - 1) / σ

/-! ## The group law read on the slots -/

/-- The first slot of the sum of two flows. -/
theorem C_add (σ s t : ℝ) :
    C σ (s + t) = C σ s * C σ t + σ * S σ s * S σ t := by
  have h := congrArg Prod.fst (Apeiron.Trichotomy.flow_add σ s t)
  simpa only [C, S, Apeiron.Trichotomy.mul, Apeiron.Trichotomy.AnyRing.ringMul] using h.symm

/-- The second slot of the sum of two flows. -/
theorem S_add (σ s t : ℝ) :
    S σ (s + t) = C σ s * S σ t + S σ s * C σ t := by
  have h := congrArg Prod.snd (Apeiron.Trichotomy.flow_add σ s t)
  simpa only [C, S, Apeiron.Trichotomy.mul, Apeiron.Trichotomy.AnyRing.ringMul] using h.symm

/-! ## Parity

`C` is even and `S` is odd at every sigma: the parity all three faces share. -/

/-- The first slot is even in time, at every sigma. -/
theorem C_neg (σ t : ℝ) : C σ (-t) = C σ t := by
  rcases lt_trichotomy σ 0 with h | h | h
  · change (Apeiron.Trichotomy.flow σ (-t)).1 = (Apeiron.Trichotomy.flow σ t).1
    rw [Apeiron.Trichotomy.flow_neg h (-t), Apeiron.Trichotomy.flow_neg h t]
    change Real.cos (Real.sqrt (-σ) * (-t)) = Real.cos (Real.sqrt (-σ) * t)
    rw [show Real.sqrt (-σ) * (-t) = -(Real.sqrt (-σ) * t) by ring, Real.cos_neg]
  · subst h
    change (Apeiron.Trichotomy.flow 0 (-t)).1 = (Apeiron.Trichotomy.flow 0 t).1
    rw [Apeiron.Trichotomy.flow_at_zero (-t), Apeiron.Trichotomy.flow_at_zero t]
  · change (Apeiron.Trichotomy.flow σ (-t)).1 = (Apeiron.Trichotomy.flow σ t).1
    rw [Apeiron.Trichotomy.flow_pos h (-t), Apeiron.Trichotomy.flow_pos h t]
    change Real.cosh (Real.sqrt σ * (-t)) = Real.cosh (Real.sqrt σ * t)
    rw [show Real.sqrt σ * (-t) = -(Real.sqrt σ * t) by ring, Real.cosh_neg]

/-- The second slot is odd in time, at every sigma. -/
theorem S_neg (σ t : ℝ) : S σ (-t) = -S σ t := by
  rcases lt_trichotomy σ 0 with h | h | h
  · change (Apeiron.Trichotomy.flow σ (-t)).2 = -(Apeiron.Trichotomy.flow σ t).2
    rw [Apeiron.Trichotomy.flow_neg h (-t), Apeiron.Trichotomy.flow_neg h t]
    change Real.sin (Real.sqrt (-σ) * (-t)) / Real.sqrt (-σ)
      = -(Real.sin (Real.sqrt (-σ) * t) / Real.sqrt (-σ))
    rw [show Real.sqrt (-σ) * (-t) = -(Real.sqrt (-σ) * t) by ring, Real.sin_neg]
    ring
  · subst h
    change (Apeiron.Trichotomy.flow 0 (-t)).2 = -(Apeiron.Trichotomy.flow 0 t).2
    rw [Apeiron.Trichotomy.flow_at_zero (-t), Apeiron.Trichotomy.flow_at_zero t]
  · change (Apeiron.Trichotomy.flow σ (-t)).2 = -(Apeiron.Trichotomy.flow σ t).2
    rw [Apeiron.Trichotomy.flow_pos h (-t), Apeiron.Trichotomy.flow_pos h t]
    change Real.sinh (Real.sqrt σ * (-t)) / Real.sqrt σ
      = -(Real.sinh (Real.sqrt σ * t) / Real.sqrt σ)
    rw [show Real.sqrt σ * (-t) = -(Real.sqrt σ * t) by ring, Real.sinh_neg]
    ring

/-! ## The defining relation of `K` -/

/-- `C` is one plus sigma times `K`, at every sigma. -/
theorem C_eq_one_add (σ t : ℝ) : C σ t = 1 + σ * K σ t := by
  by_cases h : σ = 0
  · subst h
    have hC : C 0 t = 1 := by
      simp only [C, Apeiron.Trichotomy.flow_at_zero]
    rw [hC]
    simp only [K, if_true]
    ring
  · simp only [K]
    rw [if_neg h]
    field_simp
    ring

/-! ## Derivatives

`C' = σ·S` and `S' = C` are `hasDerivAt_flow_fst`/`hasDerivAt_flow_snd` read on the slots;
`K' = S` follows from `C' = σ·S` off zero and from `(t²/2)' = t` at zero. -/

/-- The first slot's rate of change is sigma times the second slot. -/
theorem hasDerivAt_C (σ t : ℝ) : HasDerivAt (C σ) (σ * S σ t) t := by
  have h := Apeiron.Trichotomy.hasDerivAt_flow_fst σ t
  simpa only [C, S] using h

/-- The second slot's rate of change is the first slot. -/
theorem hasDerivAt_S (σ t : ℝ) : HasDerivAt (S σ) (C σ t) t := by
  have h := Apeiron.Trichotomy.hasDerivAt_flow_snd σ t
  simpa only [C, S] using h

/-- The second integral's rate of change is the second slot. -/
theorem hasDerivAt_K (σ t : ℝ) : HasDerivAt (K σ) (S σ t) t := by
  by_cases h : σ = 0
  · subst h
    have hK : K 0 = fun u : ℝ => u ^ 2 / 2 := by
      funext u
      simp only [K, if_true]
    have hS : S 0 t = t := by
      simp only [S, Apeiron.Trichotomy.flow_at_zero]
    rw [hK, hS]
    have h1 : HasDerivAt (fun u : ℝ => u) (1 : ℝ) t := hasDerivAt_id t
    have ha : HasDerivAt (fun u : ℝ => u * u) (1 * t + t * 1) t := h1.mul h1
    have hb : HasDerivAt (fun u : ℝ => u * u) (t + t) t := ha.congr_deriv (by ring)
    have hc : HasDerivAt (fun u : ℝ => u * u / 2) ((t + t) / 2) t := hb.div_const 2
    have hd : HasDerivAt (fun u : ℝ => u * u / 2) t t := hc.congr_deriv (by ring)
    convert hd using 1
    ext u
    ring
  · have hK : K σ = fun u : ℝ => (C σ u - 1) / σ := by
      funext u
      simp only [K]
      rw [if_neg h]
    rw [hK]
    have h1 : HasDerivAt (fun u : ℝ => C σ u - 1) (σ * S σ t) t :=
      (hasDerivAt_C σ t).sub_const 1
    have h2 : HasDerivAt (fun u : ℝ => (C σ u - 1) / σ) ((σ * S σ t) / σ) t :=
      h1.div_const σ
    exact h2.congr_deriv (mul_div_cancel_left₀ (S σ t) h)

/-! ## The shift identities -/

/-- The linear combination whose central differences we take. -/
abbrev f (σ a b c θ : ℝ) : ℝ := a + b * S σ θ + c * K σ θ

/-- The first derivative of `f`, `f1 = b·C + c·S`. -/
abbrev f1 (σ b c θ : ℝ) : ℝ := b * C σ θ + c * S σ θ

/-- The second derivative of `f`, `f2 = b·σ·S + c·C`. -/
abbrev f2 (σ b c θ : ℝ) : ℝ := b * σ * S σ θ + c * C σ θ

/-- The derivative of `f` is `f1`. -/
theorem hasDerivAt_f (σ a b c θ : ℝ) : HasDerivAt (f σ a b c) (f1 σ b c θ) θ := by
  have h : HasDerivAt (fun u : ℝ => a + b * S σ u + c * K σ u)
      (b * C σ θ + c * S σ θ) θ := by
    have h0 := ((hasDerivAt_const θ a).add ((hasDerivAt_S σ θ).const_mul b)).add
      ((hasDerivAt_K σ θ).const_mul c)
    exact h0.congr_deriv (by ring)
  exact h

/-- The derivative of `f1` is `f2`. -/
theorem hasDerivAt_f1 (σ b c θ : ℝ) : HasDerivAt (f1 σ b c) (f2 σ b c θ) θ := by
  have h : HasDerivAt (fun u : ℝ => b * C σ u + c * S σ u) (b * σ * S σ θ + c * C σ θ) θ := by
    have h0 := ((hasDerivAt_C σ θ).const_mul b).add ((hasDerivAt_S σ θ).const_mul c)
    exact h0.congr_deriv (by ring)
  exact h

/-- `S`'s central difference is twice `C σ θ` times `S σ s`. -/
theorem S_central (σ θ s : ℝ) :
    S σ (θ + s) - S σ (θ - s) = 2 * C σ θ * S σ s := by
  have h1 : S σ (θ + s) = C σ θ * S σ s + S σ θ * C σ s := S_add σ θ s
  have h2 : S σ (θ - s) = C σ θ * S σ (-s) + S σ θ * C σ (-s) := by
    rw [sub_eq_add_neg]
    exact S_add σ θ (-s)
  rw [h1, h2, S_neg σ s, C_neg σ s]
  ring

/-- `C`'s central difference is `2σ` times `S σ θ` times `S σ s`. -/
theorem C_central (σ θ s : ℝ) :
    C σ (θ + s) - C σ (θ - s) = 2 * σ * S σ θ * S σ s := by
  have h1 : C σ (θ + s) = C σ θ * C σ s + σ * S σ θ * S σ s := C_add σ θ s
  have h2 : C σ (θ - s) = C σ θ * C σ (-s) + σ * S σ θ * S σ (-s) := by
    rw [sub_eq_add_neg]
    exact C_add σ θ (-s)
  rw [h1, h2, S_neg σ s, C_neg σ s]
  ring

/-- `K`'s central difference is twice `S σ θ` times `S σ s`. -/
theorem K_central (σ θ s : ℝ) :
    K σ (θ + s) - K σ (θ - s) = 2 * S σ θ * S σ s := by
  by_cases h : σ = 0
  · subst h
    have hK : ∀ u : ℝ, K 0 u = u ^ 2 / 2 := by
      intro u
      simp only [K, if_true]
    have hS : ∀ u : ℝ, S 0 u = u := by
      intro u
      simp only [S, Apeiron.Trichotomy.flow_at_zero]
    rw [hK (θ + s), hK (θ - s), hS θ, hS s]
    ring
  · have hK : ∀ u : ℝ, K σ u = (C σ u - 1) / σ := by
      intro u
      simp only [K]
      rw [if_neg h]
    rw [hK (θ + s), hK (θ - s), ← sub_div]
    rw [show C σ (θ + s) - 1 - (C σ (θ - s) - 1) = C σ (θ + s) - C σ (θ - s) by ring,
      C_central σ θ s]
    field_simp

/-- `S`'s second central difference is `2σ·K σ s` times `S σ θ`. -/
theorem S_second (σ θ s : ℝ) :
    S σ (θ + s) + S σ (θ - s) - 2 * S σ θ = 2 * σ * K σ s * S σ θ := by
  have h1 : S σ (θ + s) = C σ θ * S σ s + S σ θ * C σ s := S_add σ θ s
  have h2 : S σ (θ - s) = C σ θ * S σ (-s) + S σ θ * C σ (-s) := by
    rw [sub_eq_add_neg]
    exact S_add σ θ (-s)
  rw [h1, h2, S_neg σ s, C_neg σ s, C_eq_one_add σ s]
  ring

/-- `C`'s second central difference is `2σ·K σ s` times `C σ θ`. -/
theorem C_second (σ θ s : ℝ) :
    C σ (θ + s) + C σ (θ - s) - 2 * C σ θ = 2 * σ * K σ s * C σ θ := by
  have h1 : C σ (θ + s) = C σ θ * C σ s + σ * S σ θ * S σ s := C_add σ θ s
  have h2 : C σ (θ - s) = C σ θ * C σ (-s) + σ * S σ θ * S σ (-s) := by
    rw [sub_eq_add_neg]
    exact C_add σ θ (-s)
  rw [h1, h2, C_neg σ s, S_neg σ s, C_eq_one_add σ s]
  ring

/-- `K`'s second central difference is `2·K σ s` times `C σ θ`. -/
theorem K_second (σ θ s : ℝ) :
    K σ (θ + s) + K σ (θ - s) - 2 * K σ θ = 2 * K σ s * C σ θ := by
  by_cases h : σ = 0
  · subst h
    have hK : ∀ u : ℝ, K 0 u = u ^ 2 / 2 := by
      intro u
      simp only [K, if_true]
    have hC : C 0 θ = 1 := by
      simp only [C, Apeiron.Trichotomy.flow_at_zero]
    rw [hK (θ + s), hK (θ - s), hK θ, hK s, hC]
    ring
  · have hK : ∀ u : ℝ, K σ u = (C σ u - 1) / σ := by
      intro u
      simp only [K]
      rw [if_neg h]
    have hL : K σ (θ + s) + K σ (θ - s) - 2 * K σ θ
        = (C σ (θ + s) - 1) / σ + (C σ (θ - s) - 1) / σ
            - 2 * ((C σ θ - 1) / σ) := by
      rw [hK (θ + s), hK (θ - s), hK θ]
    rw [hL]
    rw [← add_div, ← mul_div_assoc, ← sub_div]
    rw [show C σ (θ + s) - 1 + (C σ (θ - s) - 1) - 2 * (C σ θ - 1)
          = C σ (θ + s) + C σ (θ - s) - 2 * C σ θ by ring, C_second σ θ s]
    field_simp

/-- **First shift identity, uniform in sigma.**
    `f (θ + s) − f (θ − s) = 2·S σ s · f1 θ`. -/
theorem shift_first (σ a b c θ s : ℝ) :
    f σ a b c (θ + s) - f σ a b c (θ - s) = 2 * S σ s * f1 σ b c θ := by
  calc f σ a b c (θ + s) - f σ a b c (θ - s)
      = b * (S σ (θ + s) - S σ (θ - s)) + c * (K σ (θ + s) - K σ (θ - s)) := by
        simp only [f]
        ring
    _ = b * (2 * C σ θ * S σ s) + c * (2 * S σ θ * S σ s) := by
        rw [S_central σ θ s, K_central σ θ s]
    _ = 2 * S σ s * f1 σ b c θ := by
        simp only [f1]
        ring

/-- **Second shift identity, uniform in sigma.**
    `f (θ + s) + f (θ − s) − 2·f θ = 2·K σ s · f2 θ`. -/
theorem shift_second (σ a b c θ s : ℝ) :
    f σ a b c (θ + s) + f σ a b c (θ - s) - 2 * f σ a b c θ
      = 2 * K σ s * f2 σ b c θ := by
  calc f σ a b c (θ + s) + f σ a b c (θ - s) - 2 * f σ a b c θ
      = b * (S σ (θ + s) + S σ (θ - s) - 2 * S σ θ)
          + c * (K σ (θ + s) + K σ (θ - s) - 2 * K σ θ) := by
        simp only [f]
        ring
    _ = b * (2 * σ * K σ s * S σ θ) + c * (2 * K σ s * C σ θ) := by
        rw [S_second σ θ s, K_second σ θ s]
    _ = 2 * K σ s * f2 σ b c θ := by
        simp only [f2]
        ring

/-! ## The three faces -/

/-- At sigma `−1`, time `π/2`, the second slot is one. -/
theorem S_neg_one_half_pi : S (-1) (π / 2) = 1 := by
  simp only [S, Apeiron.Trichotomy.flow_neg_one]
  exact Real.sin_pi_div_two

/-- At sigma `−1`, time `π/2`, the second integral is one. -/
theorem K_neg_one_half_pi : K (-1) (π / 2) = 1 := by
  have hne : (-1 : ℝ) ≠ 0 := by norm_num
  have hC : C (-1) (π / 2) = 0 := by
    simp only [C, Apeiron.Trichotomy.flow_neg_one]
    exact Real.cos_pi_div_two
  simp only [K]
  rw [if_neg hne, hC]
  norm_num

/-- **Parameter-shift rule (elliptic face).** At `σ = −1`, `s = π/2`:
    `S (-1) (π/2) = 1`, `K (-1) (π/2) = 1`, and
    `f1 θ = (f (θ + π/2) − f (θ − π/2)) / 2`. -/
theorem parameter_shift_elliptic :
    S (-1) (π / 2) = 1 ∧ K (-1) (π / 2) = 1
      ∧ (∀ a b c θ : ℝ,
          f1 (-1) b c θ = (f (-1) a b c (θ + π / 2) - f (-1) a b c (θ - π / 2)) / 2) := by
  refine ⟨S_neg_one_half_pi, K_neg_one_half_pi, ?_⟩
  intro a b c θ
  have h := shift_first (-1) a b c θ (π / 2)
  rw [S_neg_one_half_pi] at h
  linarith

/-- **Central difference (parabolic face).** At `σ = 0`:
    `f (θ + s) − f (θ − s) = 2·s·f1 θ` and
    `f (θ + s) + f (θ − s) − 2·f θ = s²·f2 θ`, exact for the quadratics `f 0`. -/
theorem central_difference_parabolic :
    (∀ a b c θ s : ℝ, f 0 a b c (θ + s) - f 0 a b c (θ - s) = 2 * s * f1 0 b c θ)
      ∧ (∀ a b c θ s : ℝ,
          f 0 a b c (θ + s) + f 0 a b c (θ - s) - 2 * f 0 a b c θ = s ^ 2 * f2 0 b c θ) := by
  have hS0 : ∀ u : ℝ, S 0 u = u := by
    intro u
    simp only [S, Apeiron.Trichotomy.flow_at_zero]
  have hK0 : ∀ u : ℝ, K 0 u = u ^ 2 / 2 := by
    intro u
    simp only [K, if_true]
  refine ⟨?_, ?_⟩
  · intro a b c θ s
    have h := shift_first 0 a b c θ s
    rw [hS0 s] at h
    exact h
  · intro a b c θ s
    have h := shift_second 0 a b c θ s
    rw [hK0 s] at h
    rw [h]
    ring

/-- At sigma `1`, the second slot is the hyperbolic sine. -/
theorem S_one (s : ℝ) : S 1 s = Real.sinh s := by
  simp only [S, Apeiron.Trichotomy.flow_one]

/-- At sigma `1`, the second integral is `cosh − 1`. -/
theorem K_one (s : ℝ) : K 1 s = Real.cosh s - 1 := by
  have hne : (1 : ℝ) ≠ 0 := one_ne_zero
  have hC : C 1 s = Real.cosh s := by
    simp only [C, Apeiron.Trichotomy.flow_one]
  simp only [K]
  rw [if_neg hne, hC]
  ring

/-- **Hyperbolic shift (hyperbolic face).** At `σ = 1`: `S 1 s = sinh s`,
    `K 1 s = cosh s − 1`, and both shift identities read
    `2·sinh s · f1 θ` and `2·(cosh s − 1) · f2 θ`. -/
theorem shift_hyperbolic :
    (∀ s : ℝ, S 1 s = Real.sinh s) ∧ (∀ s : ℝ, K 1 s = Real.cosh s - 1)
      ∧ (∀ a b c θ s : ℝ,
          f 1 a b c (θ + s) - f 1 a b c (θ - s) = 2 * Real.sinh s * f1 1 b c θ)
      ∧ (∀ a b c θ s : ℝ,
          f 1 a b c (θ + s) + f 1 a b c (θ - s) - 2 * f 1 a b c θ
            = 2 * (Real.cosh s - 1) * f2 1 b c θ) := by
  refine ⟨S_one, K_one, ?_, ?_⟩
  · intro a b c θ s
    have h := shift_first 1 a b c θ s
    rw [S_one s] at h
    exact h
  · intro a b c θ s
    have h := shift_second 1 a b c θ s
    rw [K_one s] at h
    exact h

/-! ## The boundary: a cubic is outside the span

`g θ = θ³` has `g (θ+s) − g (θ−s) = 6θ²s + 2s³`, so `shift_first` with `f1 θ = 3θ²` fails at
`θ = 0`, `s = 1`. The pair of shift identities therefore spans the flow's own combinations and
their quadratics, not every function. -/

/-- The cubic `θ³` violates the first shift identity. -/
theorem cubic_violates_shift_first :
    ¬ (∀ θ s : ℝ, (θ + s) ^ 3 - (θ - s) ^ 3 = 2 * s * (3 * θ ^ 2)) := by
  intro h
  have h1 := h 0 1
  norm_num at h1

/-! ## Verdict -/

/-- The unit's verdict: both shift identities hold uniformly in sigma, and a cubic
    counterexample shows the span of `shift_first` is not all functions. -/
theorem verdict_l5 :
    (∀ σ a b c θ s : ℝ, f σ a b c (θ + s) - f σ a b c (θ - s) = 2 * S σ s * f1 σ b c θ)
      ∧ (∀ σ a b c θ s : ℝ,
          f σ a b c (θ + s) + f σ a b c (θ - s) - 2 * f σ a b c θ
            = 2 * K σ s * f2 σ b c θ)
      ∧ ¬ (∀ θ s : ℝ, (θ + s) ^ 3 - (θ - s) ^ 3 = 2 * s * (3 * θ ^ 2)) :=
  ⟨shift_first, shift_second, cubic_violates_shift_first⟩

end FaceShift
end Quantum
