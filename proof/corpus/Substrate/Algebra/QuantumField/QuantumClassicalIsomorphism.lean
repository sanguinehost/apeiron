import Substrate.Algebra.QuantumField.MadelungCell
import Substrate.Algebra.NullCell.NullCell
import Substrate.Algebra.QuantumField.WickRotation
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Tactic.Common
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# The Quantum–Classical Isomorphism — classical action + K=4 branching = the quantum wave

@name Building a Quantum Wave From Four Classical Paths
@child An ordinary moving ball follows one path. A quantum wave needs more — it needs to feel how its height curves, which one path can't see. Sample four paths arranged like the four points of a compass, though, and the curve comes back exactly.
@scientist The quantum wave is recoverable from the classical action up to one missing term, the quantum potential ∇²R/R; a single classical branch (value + gradient) can't supply a Laplacian, but the four 4th-roots-of-unity branches Iᵏ = {1, I, −1, −I} form a symmetric stencil that reconstructs it exactly for the local amplitude — the same K = 4 = ℍ the evaluator forces.
@engineer `quantum_potential_is_the_gap` locates the missing term; `single_branch_underdetermines` shows one path fails; `stencil4_reconstructs_laplacian` + `four_branches_are_rotor_orbit` recover ∇²R, with `branches_wick_dual_to_classical` welding it to the Null-cell classical action via Wick rotation.

The one final bridge. Slotine–Lohmiller (ref23, *On computing quantum waves exactly from
classical action*) showed the quantum wave is recoverable from the classical action; Vattay
(ref24) sharpened it: the recovery is exact **only with spatial dispersion / multi-branching**,
because the quantum potential `Q ∝ ∇²R/R` carries the amplitude's *second* spatial derivative,
which a single classical branch (a point trajectory: value + momentum) cannot supply. This file
reproduces both, then closes the gap with **K = 4 branching** — and welds the result to the
`NullCell` basis.

## The three movements

1. **Slotine–Lohmiller — the gap IS the quantum potential.** Reusing `MadelungCell.cellCurv_re`,
   the Madelung real channel `Re(ψ'')/R = R''/R − (φ')²` minus the *classical* Hamilton–Jacobi
   kinetic term `−(φ')²` is exactly `R''/R` (`quantum_potential_is_the_gap`). So the classical
   action computes the wave up to one term, and that term is the bulk ratio `∇²R/R` — the
   quantum potential of `MadelungCell`.

2. **Vattay — one branch cannot.** A single classical branch carries the amplitude's *value and
   gradient* at a point (`jet1`); these do **not** determine its Laplacian —
   `single_branch_underdetermines` is the non-existence, `¬ ∃ F, ∀ …, F (jet1 …) = p + q`, because
   two quadratics with the same 1-jet at the origin differ in `∇²R`. The quantum potential is
   invisible to one trajectory — the dispersion must be sampled.

3. **K = 4 — the rotor orbit reconstructs it exactly.** The four branches are the **4th roots of
   unity** `Iᵏ = {1, I, −1, −I}` (`four_branches_are_rotor_orbit`, period 4 — the Being-cell
   rotor `i²=−1` turned a quarter-turn four times), the four cardinal directions of the phase
   cell `ℂ`. The 4-branch symmetric stencil over them reconstructs the Laplacian **exactly** for
   the local (quadratic) amplitude (`stencil4_reconstructs_laplacian`: `Δ₄R = ∇²R = p + q`, the
   linear and cross terms cancelling by the rotor symmetry). K=4 is the minimal symmetric stencil
   that recovers the isotropic 2-D Laplacian — the same `K=4=ℍ` the framework's evaluator and
   `DivisionAlgebraSubstrate` already force.

## The Wick weld (the `NullCell` basis)

The four branches live in the **Being cell** (`I² = −1`, the rotor — `branch_is_being_cell`); the
classical action lives in the **Null cell** (`j² = +1`, the boost — `NullCell.nj_sq`). The Wick
rotation `i² = −1 ↔ j² = +1` (`WickRotation`, `branches_wick_dual_to_classical`) is the
classical↔quantum bridge: un-Wick the classical Null-cell action, sample its dispersion on the
K=4 Being-cell rotor orbit, and the quantum potential is restored. *The arrow of time is not a
wall here — it is the Wick rotation, a chart swap; the branches are Everett superposition made
classical.*

Cited: the `−ℏ²/2m` Schrödinger prefactor (as in `MadelungCell`) and the all-orders
(non-quadratic) reconstruction as the `h → 0` finite-difference limit; that the framework's K=4
evaluator branching *is* this 4-point dispersion stencil is the interpreted identification on top
of the proved stencil.

## What the isomorphism warrants

* **Proved (native):** `stencil4_reconstructs_laplacian` and `antithetic_recovers_gradient` — the
  two recoveries, with `orbit_mean_is_value_plus_curvature` the K=1 tier and
  `single_branch_underdetermines` the obstruction they answer. For `quad R0 a b p q c`, the symmetric 4-point stencil is
  exactly `p + q` and the antithetic difference exactly `a + p·x + c·y`, for every `h ≠ 0`, with the
  odd/even cancellation done by `field_simp; ring`. `secondDiff1D` is the 1-D anchor.
* **Proved (native):** `quantum_potential_is_the_gap` — `MadelungCell.cellCurv_re` normalised by
  `R y`, minus `classicalHJ`, equals `ddR y / R y`. Real, and the only place `MadelungCell` is
  actually used.
* **Proved (Mathlib):** `four_branches_are_rotor_orbit`, `stencil_offsets_are_the_orbit` and
  `branch_is_being_cell` are `Complex.I_sq`, `pow_succ` and `interval_cases` over `Finset.range 4`.
* **Proved (native):** Vattay's obstruction. `single_branch_underdetermines` is now
  `¬ ∃ F : ℝ × ℝ × ℝ → ℝ, ∀ R0 a b p q c, F (jet1 (quad R0 a b p q c) h) = p + q` — the Hessian
  trace is not a FUNCTION of the 1-jet, with `jet1` the value and the two antithetic partials and
  `jet1_quad` proving that triple is exactly `(R₀, a, b)` for every `h ≠ 0`. The witnesses
  `quad 0 0 0 0 0 0` and `quad 0 0 0 1 1 0` share the jet `(0,0,0)` and differ at `0 + 0 ≠ 1 + 1`.
  Strengthened 2026-08-05 from `∃ R0 a b c p q p' q', quad R0 a b p q c 0 0 = quad R0 a b p' q' c 0 0
  ∧ (a = a ∧ b = b) ∧ p + q ≠ p' + q'`, whose first conjunct held for all arguments, whose second
  was `⟨rfl, rfl⟩` between syntactically identical terms, and whose content was `0 + 0 ≠ 1 + 1`.
  It is conjunct 2 of the headline, which previously omitted it.
* **Proved (native):** `spinor_subtrinity`'s first conjunct is now
  `orbit_mean_is_value_plus_curvature` — the antithetic MEAN is `quad … x y + (p/2)·h²`, the value
  plus exactly the curvature the orbit encloses, so the value alone at radius `0`. Replaced
  2026-08-05; it was `quad R0 a b p q c x y = quad R0 a b p q c x y` by `rfl`, the textual `X = X`
  `THE_ONTOLOGY.md` §5 names as R11's text-decidable vacuity. The other two conjuncts are
  `antithetic_recovers_gradient` and `stencil4_reconstructs_laplacian`, both real and untouched.
* **Proved (riding):** "Wick dual". `branches_wick_dual_to_classical` now carries the MAP:
  `wick b = b·I` transports the classical Null interval to the branch Being interval on every
  `NCell` (`WickRotation.nNorm_wick_to_phase`), the four branches are the orbit of `wick 1`
  (`stencil_offsets_are_the_orbit`), and the sign flip is `wick 1 ^ 2 = -1` against
  `NullCell.nj_sq`. Strengthened 2026-08-05 from `Complex.I ^ 2 = -1 ∧ nmul nj nj = nOne`, two
  independent facts side by side with no map, in which no `WickRotation` declaration occurred
  though the module was imported.
* **Open:** minimality — *"K=4 is the minimal symmetric stencil that recovers the
  isotropic 2-D Laplacian."* No stencil other than `stencil4` and the antithetic pair is defined,
  and no minimality statement exists.
* **Anchor:** the Slotine–Lohmiller (ref23) and Vattay (ref24) attributions. Both are
  published results and that is the outside evidence, but the referenced `references.bib` is **not
  in this repository** — the only file of that name on disk is Mathlib's own
  (`proof/.lake/packages/mathlib/the bibliography), so `ref23`/`ref24` currently resolve to
  nothing.
* **Anchor:** the K=1/2/4 tiers as spinor matrix sizes `1×1/2×2/4×4` and as
  `{ℂ, ℍ, 𝕆}`. `spinor_subtrinity`'s own docstring already labels this a bridge, correctly; the
  supporting artefacts `SpinorLoop.lean`, `TrinityCodex.division_face_trinity`,
  `DivisionAlgebraSubstrate.lean` and `theory/THE_HASHMAP_PATH.md` all resolve. No statement here
  relates a stencil to a matrix size.
* **Open:** exactness is exact for `quad` only. `stencil4_reconstructs_laplacian` is a polynomial
  identity on the quadratic model; the all-orders reconstruction as an `h → 0` limit is named in
  the header as cited and is proved nowhere. `classicalHJ` is a definition, not a derived classical
  action.

-/

noncomputable section

open Complex

namespace Sanguine
namespace QuantumClassicalIsomorphism

-- ═══════════════════════════════════════════════════════════════════
-- §1. SLOTINE–LOHMILLER — THE QUANTUM-CLASSICAL GAP IS THE QUANTUM POTENTIAL
-- ═══════════════════════════════════════════════════════════════════

/-- The **classical** Hamilton–Jacobi real channel — the kinetic term alone, `−(φ')²`. The
    amplitude does not disperse: the `R''` (quantum-potential) term is dropped. This is what a
    classical action computes. -/
def classicalHJ (dφ : ℝ → ℝ) (y : ℝ) : ℝ := -(dφ y) ^ 2

/-- **The quantum potential IS the quantum−classical gap (Slotine–Lohmiller).** The Madelung real
    channel (`MadelungCell.cellCurv_re`, normalized by `R`) minus the classical HJ kinetic term is
    exactly the bulk ratio `R''/R` — the quantum potential. The classical action gets you the wave
    up to this one term, and the term is `∇²R/R`. -/
theorem quantum_potential_is_the_gap (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) (hR : R y ≠ 0) :
    (Sanguine.MadelungCell.cellCurv R dR ddR dφ ddφ y).re / R y - classicalHJ dφ y
      = ddR y / R y := by
  rw [MadelungCell.cellCurv_re]
  simp only [classicalHJ]
  field_simp
  ring

-- ═══════════════════════════════════════════════════════════════════
-- §2. VATTAY — ONE CLASSICAL BRANCH CANNOT RECOVER THE QUANTUM POTENTIAL
-- ═══════════════════════════════════════════════════════════════════

/-- The local **quadratic amplitude model** `R(x,y) = R₀ + a·x + b·y + ½p·x² + ½q·y² + c·xy`.
    Its quantum potential (Laplacian) is the constant `p + q` (the Hessian trace). The generic
    smooth amplitude is this to second order — the order the quantum potential lives at. -/
def quad (R0 a b p q c x y : ℝ) : ℝ :=
  R0 + a * x + b * y + (p / 2) * x ^ 2 + (q / 2) * y ^ 2 + c * x * y

/-- **The 1-D second difference IS the second derivative.** For `g(x) = α + βx + ½p·x²`,
    `(g(x+h) − 2g(x) + g(x−h))/h² = p = g''` — the anchor that the symmetric stencil computes
    curvature, not a value. -/
theorem secondDiff1D (α β p x h : ℝ) (hh : h ≠ 0) :
    (quad α β 0 p 0 0 (x + h) 0 - 2 * quad α β 0 p 0 0 x 0 + quad α β 0 p 0 0 (x - h) 0) / h ^ 2
      = p := by
  simp only [quad]; field_simp; ring

/-- The **1-jet** a single classical branch carries at the origin: the value, and the two partials
    read by the antithetic (K=2) orbit at step `h` — position and momentum, and nothing else. -/
def jet1 (f : ℝ → ℝ → ℝ) (h : ℝ) : ℝ × ℝ × ℝ :=
  (f 0 0, (f h 0 - f (-h) 0) / (2 * h), (f 0 h - f 0 (-h)) / (2 * h))

/-- For the local quadratic the 1-jet is exactly `(R₀, a, b)`, at every step `h ≠ 0` — the
    antithetic orbit reads the gradient exactly (`antithetic_recovers_gradient` at the origin), so
    the branch datum is that triple and no more. -/
theorem jet1_quad (R0 a b p q c h : ℝ) (hh : h ≠ 0) :
    jet1 (quad R0 a b p q c) h = (R0, a, b) := by
  simp only [jet1, quad, Prod.mk.injEq]
  refine ⟨by ring, ?_, ?_⟩ <;> (field_simp; ring)

/-- **Vattay's obstruction — the Hessian trace is not a function of the 1-jet.** There is NO
    `F` from the branch datum `(value, ∂ₓ, ∂_y)` to the Laplacian `p + q`: two quadratics with the
    same 1-jet at the origin have different `p + q`, so any such `F` would have to return two
    values at one argument. The quantum potential is invisible to one branch — the dispersion must
    be sampled from more than one, which §4's `stencil4_reconstructs_laplacian` then does. -/
theorem single_branch_underdetermines (h : ℝ) (hh : h ≠ 0) :
    ¬ ∃ F : ℝ × ℝ × ℝ → ℝ, ∀ R0 a b p q c : ℝ, F (jet1 (quad R0 a b p q c) h) = p + q := by
  rintro ⟨F, hF⟩
  have h₀ := hF 0 0 0 0 0 0
  have h₁ := hF 0 0 0 1 1 0
  rw [jet1_quad 0 0 0 0 0 0 h hh] at h₀
  rw [jet1_quad 0 0 0 1 1 0 h hh] at h₁
  rw [h₀] at h₁
  norm_num at h₁

-- ═══════════════════════════════════════════════════════════════════
-- §3. K=4 — THE FOUR BRANCHES ARE THE 4th ROOTS OF UNITY (THE ROTOR ORBIT)
-- ═══════════════════════════════════════════════════════════════════

/-- **The four branches are the rotor orbit** `Iᵏ = {1, I, −1, −I}`, period 4. The Being-cell
    rotor `i² = −1` turned a quarter-turn, closing after four — the K=4 of the phase cell, the
    four cardinal directions `{±1, ±I}` the stencil samples. -/
theorem four_branches_are_rotor_orbit :
    Complex.I ^ 0 = 1 ∧ Complex.I ^ 1 = Complex.I ∧ Complex.I ^ 2 = -1
      ∧ Complex.I ^ 3 = -Complex.I ∧ Complex.I ^ 4 = 1 := by
  refine ⟨pow_zero _, pow_one _, Complex.I_sq, ?_, ?_⟩
  · rw [pow_succ, Complex.I_sq]; ring
  · rw [show (4 : ℕ) = 2 + 2 from rfl, pow_add, Complex.I_sq]; ring

/-- The stencil's four offset directions `{1, −1, I, −I}` are exactly the four powers `Iᵏ`,
    `k < 4` — the rotor orbit as a set. The K=4 branch count is `|⟨I⟩| = 4`. -/
theorem stencil_offsets_are_the_orbit :
    (Finset.image (fun k => Complex.I ^ k) (Finset.range 4)) = {1, Complex.I, -1, -Complex.I} := by
  ext z
  simp only [Finset.mem_image, Finset.mem_range, Finset.mem_insert, Finset.mem_singleton]
  constructor
  · rintro ⟨k, hk, rfl⟩
    interval_cases k
    · exact Or.inl (pow_zero _)
    · exact Or.inr (Or.inl (pow_one _))
    · exact Or.inr (Or.inr (Or.inl Complex.I_sq))
    · exact Or.inr (Or.inr (Or.inr (by rw [pow_succ, Complex.I_sq]; ring)))
  · rintro (rfl | rfl | rfl | rfl)
    · exact ⟨0, by norm_num⟩
    · exact ⟨1, by norm_num⟩
    · exact ⟨2, by norm_num, Complex.I_sq⟩
    · exact ⟨3, by norm_num, by rw [pow_succ, Complex.I_sq]; ring⟩

-- ═══════════════════════════════════════════════════════════════════
-- §4. K=4 RECONSTRUCTS THE QUANTUM POTENTIAL — THE 4-BRANCH STENCIL
-- ═══════════════════════════════════════════════════════════════════

/-- **The K=4 dispersion stencil.** Sample the amplitude at the central trajectory and its four
    rotor-orbit branches `(x±h, y)`, `(x, y±h)`, take the symmetric second difference. This is the
    discrete Laplacian over the four branches — the dispersion Vattay says is required. -/
def stencil4 (f : ℝ → ℝ → ℝ) (x y h : ℝ) : ℝ :=
  (f (x + h) y + f (x - h) y + f x (y + h) + f x (y - h) - 4 * f x y) / h ^ 2

/-- **K=4 reconstructs the quantum potential exactly (the close).** For the local quadratic
    amplitude, the 4-branch stencil equals the Laplacian `∇²R = p + q` — **exactly**, for every
    step `h ≠ 0`. The linear terms (`a·x`, `b·y`) cancel by the symmetry of the `±` branches, and
    the cross term `c·xy` cancels because it is odd under each axis flip — the rotor symmetry of
    the four branches doing the work. So the classical amplitude, sampled on the K=4 rotor orbit,
    reconstructs the quantum potential the single branch could not (§2). -/
theorem stencil4_reconstructs_laplacian (R0 a b p q c x y h : ℝ) (hh : h ≠ 0) :
    stencil4 (quad R0 a b p q c) x y h = p + q := by
  simp only [stencil4, quad]; field_simp; ring

/-- **K=2 recovers the gradient (the order-1 sibling).** The antithetic symmetric difference along
    an axis — the `{+1, −1}` orbit — equals the exact partial derivative `∂ₓR = a + p·x + c·y` for
    the local quadratic. The two-branch orbit keeps the *odd* (gradient) term and cancels the even;
    the four-branch orbit (`stencil4`) keeps the *even* (curvature) term and cancels the odd. One
    primitive, the orbit geometry choosing the derivative order — the dispersion merge. -/
theorem antithetic_recovers_gradient (R0 a b p q c x y h : ℝ) (hh : h ≠ 0) :
    (quad R0 a b p q c (x + h) y - quad R0 a b p q c (x - h) y) / (2 * h)
      = a + p * x + c * y := by
  simp only [quad]; field_simp; ring

/-- **K=1 is the zero-radius orbit.** The MEAN over the antithetic pair is the value plus exactly
    the curvature the orbit encloses, `f x y + (p/2)·h²`. Send the radius to `0` — the single
    branch, no orbit — and the mean is the value alone. That is what "K=1 reads the value" says,
    and it is a statement about the orbit rather than an identity of a term with itself. -/
theorem orbit_mean_is_value_plus_curvature (R0 a b p q c x y h : ℝ) :
    (quad R0 a b p q c (x + h) y + quad R0 a b p q c (x - h) y) / 2
      = quad R0 a b p q c x y + (p / 2) * h ^ 2 := by
  simp only [quad]; ring

-- ═══════════════════════════════════════════════════════════════════
-- §5. THE WICK WELD — THE BRANCHES (BEING CELL) ARE DUAL TO THE CLASSICAL (NULL CELL)
-- ═══════════════════════════════════════════════════════════════════

/-- **The branches live in the Being cell** — the rotor `I² = −1`. -/
theorem branch_is_being_cell : Complex.I ^ 2 = -1 := Complex.I_sq

/-- **Imaginary time** — the Wick substitution `b ↦ b·I`, the one map between the two cells
    (`WickRotation.imaginary_time_flips_sign`). -/
def wick (b : ℝ) : ℂ := (b : ℂ) * Complex.I

/-- **The branches are the Wick dual of the classical action — and this is the map.** Imaginary
    time `wick` carries the classical Null-cell interval `a² − b²` to the branch Being-cell
    interval `a² + b²` on every `NCell` at once (`WickRotation.nNorm_wick_to_phase`); it carries
    the classical unit `1` to the branch generator, whose orbit is exactly the four branches
    (`stencil_offsets_are_the_orbit`); and the whole of the rotation is the one sign — `−1` on the
    branch side where the Null unit gives `+1` (`NullCell.nj_sq`). The duality is carried by the
    map, not by writing the two facts side by side. -/
theorem branches_wick_dual_to_classical :
    (∀ z : Sanguine.CrossUniverse.NCell,
        (z.1 : ℂ) ^ 2 - wick z.2 ^ 2 = ((z.1 ^ 2 + z.2 ^ 2 : ℝ) : ℂ))
      ∧ Finset.image (fun k => wick 1 ^ k) (Finset.range 4)
          = {1, Complex.I, -1, -Complex.I}
      ∧ (wick 1 ^ 2 = -1
          ∧ Sanguine.CrossUniverse.nmul Sanguine.CrossUniverse.nj Sanguine.CrossUniverse.nj
              = Sanguine.CrossUniverse.nOne) := by
  have hw : wick 1 = Complex.I := by simp [wick]
  refine ⟨fun z => Sanguine.CrossUniverse.nNorm_wick_to_phase z, ?_, ?_, Sanguine.CrossUniverse.nj_sq⟩
  · simp only [hw]; exact stencil_offsets_are_the_orbit
  · rw [hw]; exact Complex.I_sq

-- ═══════════════════════════════════════════════════════════════════
-- §6. THE HEADLINE — THE ISOMORPHISM
-- ═══════════════════════════════════════════════════════════════════

/-- **The quantum–classical isomorphism (headline).** Classical action + K=4 branching = the
    quantum wave:

    1. **(Slotine–Lohmiller)** the quantum−classical gap is exactly the quantum potential
       `R''/R` (`quantum_potential_is_the_gap`);
    2. **(Vattay)** a single branch — value and gradient — cannot recover it
       (`single_branch_underdetermines`);
    3. **(K=4)** the four-branch rotor-orbit stencil (`Iᵏ`, `k<4`, period 4) reconstructs the
       Laplacian `∇²R = p + q` exactly for the local amplitude (`stencil4_reconstructs_laplacian`);
    4. **(Wick)** the branches are the Being-cell rotor `I²=−1`, the classical action the
       Null-cell boost `j²=+1`, the two the Wick dual (`branches_wick_dual_to_classical`).

    The classical action does not lose the quantum information; it disperses it across the K=4
    rotor orbit, where the symmetric second difference reads it back. Time reversal is the Wick
    rotation; the branches are superposition made classical. -/
theorem quantum_classical_isomorphism
    (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) (hR : R y ≠ 0)
    (R0 a b p q c x : ℝ) (h : ℝ) (hh : h ≠ 0) :
    -- (1) the gap is the quantum potential
    ((Sanguine.MadelungCell.cellCurv R dR ddR dφ ddφ y).re / R y - classicalHJ dφ y = ddR y / R y)
      -- (2) no function of the single branch's 1-jet delivers it
      ∧ (¬ ∃ F : ℝ × ℝ × ℝ → ℝ,
            ∀ R0' a' b' p' q' c' : ℝ, F (jet1 (quad R0' a' b' p' q' c') h) = p' + q')
      -- (3) K=4 reconstructs it exactly for the local model
      ∧ (stencil4 (quad R0 a b p q c) x y h = p + q)
      -- (4) the branches are the Being↔Null Wick dual: the map, its orbit, the sign flip
      ∧ ((∀ z : Sanguine.CrossUniverse.NCell,
            (z.1 : ℂ) ^ 2 - wick z.2 ^ 2 = ((z.1 ^ 2 + z.2 ^ 2 : ℝ) : ℂ))
          ∧ Finset.image (fun k => wick 1 ^ k) (Finset.range 4)
              = {1, Complex.I, -1, -Complex.I}
          ∧ (wick 1 ^ 2 = -1
              ∧ Sanguine.CrossUniverse.nmul Sanguine.CrossUniverse.nj Sanguine.CrossUniverse.nj
                  = Sanguine.CrossUniverse.nOne)) :=
  ⟨quantum_potential_is_the_gap R dR ddR dφ ddφ y hR,
   single_branch_underdetermines h hh,
   stencil4_reconstructs_laplacian R0 a b p q c x y h hh,
   branches_wick_dual_to_classical⟩

-- ═══════════════════════════════════════════════════════════════════
-- §7. THE SPINOR SUB-TRINITY — the rotor at K=1/2/4, ONE operator
-- ═══════════════════════════════════════════════════════════════════
-- The dispersion merge IS the spinor sub-trinity: the rotor orbit `iᵏ` read at THREE tiers, the
-- spinor's three matrix sizes — 1×1, 2×2, 4×4 (`SpinorLoop.AffineStep`/`AffineStepN n` at n=1,2,4 =
-- U(1)/SU(2)/Dirac) = `TrinityCodex.division_face_trinity` `{ℂ, ℍ, 𝕆}` (the three doublings where
-- `/` stays total). There is only ONE operator (`THE_HASHMAP_PATH`: the merge / the signed orbit
-- sum / the affine `compose`); the orbit GEOMETRY (the radius / how many branches) selects which
-- derivative order survives — value, gradient, curvature. This is how `x = B/U` was first found:
-- `∇²R/R` (the curvature tier over the value tier) IS the quantum potential, the bulk ratio
-- (`MadelungCell.quantum_potential_is_bulk_ratio`).

/-- **The spinor sub-trinity — three tiers, one operator (headline).** The rotor orbit at K=1/2/4 is
    the three matrix tiers of the spinor, and the one merge reads off the derivative the orbit
    selects:

    1. **K=1 (1×1, the value / the monad tier)** — the field at the point, `f x` (no orbit, the
       single branch; the K=2/K=4 orbits are this tier dispersed);
    2. **K=2 (2×2, the gradient)** — the antithetic orbit `{+1,−1}` keeps the odd term
       (`antithetic_recovers_gradient`): `∂ₓf = a + p·x + c·y`, exact;
    3. **K=4 (4×4, the Laplacian)** — the rotor orbit `iᵏ` keeps the even term
       (`stencil4_reconstructs_laplacian`): `∇²f = p + q`, exact.

    One operator (the signed orbit sum, the affine `compose` — `SpinorLoop`), three tiers
    (`{ℂ,ℍ,𝕆}` = `division_face_trinity`), the orbit geometry the only freedom; forward-only, no
    backprop. And `∇²R/R = x = B/U` (`MadelungCell.quantum_potential_is_bulk_ratio`) — the curvature
    tier over the value tier IS the bulk equation. *(Native: the three recoveries here. Bridge: the
    identification of the orbit tiers with the spinor matrix sizes `1×1/2×2/4×4` and with `{ℂ,ℍ,𝕆}`
    — the reading `THE_HASHMAP_PATH` §the dispersion merge / §25 lays on these primitives.)* -/
theorem spinor_subtrinity (R0 a b p q c x y h : ℝ) (hh : h ≠ 0) :
    -- K=1 (value): the orbit's mean is the value plus exactly the curvature it encloses — the
    -- value alone at radius 0, the monad tier
    ((quad R0 a b p q c (x + h) y + quad R0 a b p q c (x - h) y) / 2
        = quad R0 a b p q c x y + (p / 2) * h ^ 2)
    -- K=2 (gradient): the antithetic orbit, exact
      ∧ ((quad R0 a b p q c (x + h) y - quad R0 a b p q c (x - h) y) / (2 * h) = a + p * x + c * y)
    -- K=4 (Laplacian): the rotor orbit, exact
      ∧ (stencil4 (quad R0 a b p q c) x y h = p + q) :=
  ⟨orbit_mean_is_value_plus_curvature R0 a b p q c x y h,
   antithetic_recovers_gradient R0 a b p q c x y h hh,
   stencil4_reconstructs_laplacian R0 a b p q c x y h hh⟩

end QuantumClassicalIsomorphism
end Sanguine

end
