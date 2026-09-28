import Substrate.Algebra.NullCell.NullCell
import Substrate.Algebra.Bulk.BulkTower
import Mathlib.Analysis.Complex.Trigonometric

/-!
# Wick Rotation — the Being ↔ Null Swap, and Why Naive HJ Is the Euclidean Shadow

@name The Trick That Turns Spinning Into Falling
@child There are two kinds of "stage" for physics: one where everything spins around smoothly, and one where time runs out like water down a funnel — our real world. A single trick swaps one for the other: just put an imaginary number on the clock.
@scientist Wick rotation `b ↦ i·b` swaps the definite Being cell (ℂ, i² = −1, the Euclidean rotor, norm a² + b²) for the indefinite Null cell (split-ℂ, j² = +1, the Lorentzian boost, norm a² − b²) — the one sign flip of the time-squared term that is the whole of Non-Being.
@engineer `imaginary_time_flips_sign` ((b·i)² = −b²); `wick_metric` (a² − (ib)² = a² + b²); `rotor_wick_to_boost` (cos(iφ) = cosh φ); `cell_units_swap` ties it to `Complex.I_sq` ↔ `NullCell.nj_sq`.

The Madelung / Hamilton–Jacobi picture (`MadelungCell`) lives on the **Being cell**: the
rotor `ψ = R·e^{iφ}`, `ℂ` with `i² = −1`, the definite norm `a² + b²`. It is smooth,
compact, all-spin — and it treats *time* as another phase advance (a rotor turning). That
is the "naive HJ spacetime": one spinning disk on a smooth Euclidean stage.

Real spacetime is Lorentzian: its time direction is the **Null cell** (`NullCell`), the
split-complex boost, `j² = +1`, the indefinite norm `a² − b²`, the lightcone funnel that
runs out and never returns. The single difference between the two is **the sign of the
time-squared term** — which `NullCell` proved is the whole of Non-Being.

**Wick rotation is the operation that swaps them.** The substitution `b ↦ i·b` (real time
→ imaginary time) turns the Lorentzian interval `a² − b²` into the Euclidean interval
`a² + b²` (`wick_metric`), and turns the Null-cell boost into the Being-cell rotor —
`cos(iφ) = cosh φ` (`rotor_wick_to_boost`, `Complex.cos_mul_I`). It is exactly the
`i² = −1 ↔ j² = +1` sign flip (`cell_units_swap`).

So the naive Madelung/HJ picture is the **Wick-rotated, all-Being, Euclidean shadow** of
the one-Null-cell Lorentzian world:

* **imaginary time** = all spinning disks = the smooth naive stage (the side where you
  compute the quantum potential `x = B/U = ∇²R/R`);
* **real time** = one funnel punched in = the Lorentzian spacetime we live in (the side
  with a lightcone, where `x = B/U` goes singular — Non-Being, causal structure).

Un-Wick-rotating — letting one spinning disk spring back into a lightcone funnel — is
exactly the promotion from the naive picture to the real one. The only thing the rotation
moves is the sign of the time-squared term; everything else (the amplitude `R`, which by
`AmplitudeUnification` *is* the geometry) rides along unchanged.

## What is proved vs cited

* **Proved (Mathlib):** the metric swap `a² − (ib)² = a² + b²`; the dynamical
  swap `cos(iφ) = cosh φ`, `sin(iφ) = i·sinh φ` (the rotor becomes the boost); that this
  is the `i² = −1` (`Complex.I_sq`) ↔ `j² = +1` (`NullCell.nj_sq`) sign flip.
* **Anchor:** that the corpus's Madelung/HJ picture is the Euclidean side
  and physical spacetime is the Lorentzian side, related by this one rotation — the same
  interpreted reading as the rest of the `x = B/U` physics. The standard physics name for
  this map is Wick rotation; here it is identified with the Being-cell ↔ Null-cell swap.

-/

noncomputable section

namespace Sanguine
namespace CrossUniverse

-- ═══════════════════════════════════════════════════════════════════
-- THE METRIC SWAP — IMAGINARY TIME FLIPS THE TIME-SQUARED SIGN
-- ═══════════════════════════════════════════════════════════════════

/-- **Imaginary time flips the sign.** `(b·i)² = −b²`: putting an `i` on the time
    coordinate turns its squared contribution from `+b²` (spacelike, Being) to `−b²`
    (timelike, Null). This one identity is the entire engine of Wick rotation. -/
theorem imaginary_time_flips_sign (b : ℝ) :
    ((b : ℂ) * Complex.I) ^ 2 = -(b : ℂ) ^ 2 := by
  rw [mul_pow, Complex.I_sq]; ring

/-- **Wick rotation at the metric level.** Real time gives the Lorentzian (Null-cell)
    interval `a² − b²`; imaginary time `b ↦ i·b` gives the Euclidean (phase-cell) interval
    `a² + b²`. The Lorentzian funnel is the Euclidean disk seen with imaginary time. -/
theorem wick_metric (a b : ℝ) :
    (a : ℂ) ^ 2 - ((b : ℂ) * Complex.I) ^ 2 = (a : ℂ) ^ 2 + (b : ℂ) ^ 2 := by
  rw [imaginary_time_flips_sign]; ring

/-- **The Null cell is the real-time interval.** `nNorm` (the split norm) is exactly the
    Lorentzian `a² − b²`. -/
theorem nNorm_is_real_time_interval (z : NCell) : nNorm z = z.1 ^ 2 - z.2 ^ 2 := rfl

/-- **Wick-rotating the Null cell gives the phase-cell norm.** The split norm `a² − b²`,
    under imaginary time, becomes the definite `a² + b² = ‖a + bi‖²` — the Being cell.
    The `−` of Non-Being becomes the `+` of Being; nothing else changes. -/
theorem nNorm_wick_to_phase (z : NCell) :
    (z.1 : ℂ) ^ 2 - ((z.2 : ℂ) * Complex.I) ^ 2 = ((z.1 ^ 2 + z.2 ^ 2 : ℝ) : ℂ) := by
  rw [imaginary_time_flips_sign]; push_cast; ring

-- ═══════════════════════════════════════════════════════════════════
-- THE DYNAMICAL SWAP — THE ROTOR BECOMES THE BOOST
-- ═══════════════════════════════════════════════════════════════════

/-- **Wick rotation at the dynamical level: the rotor becomes the boost.** The Being-cell
    rotor's cosine, at an imaginary angle, is the Null-cell boost's hyperbolic cosine:
    `cos(iφ) = cosh φ`; and `sin(iφ) = i·sinh φ`. The Madelung phase `e^{iφ}` (compact,
    returns at `2π`/`4π`) and the Lorentz boost (non-compact, runs out along the
    lightcone) are one structure under `θ = iφ`. -/
theorem rotor_wick_to_boost (φ : ℝ) :
    Complex.cos ((φ : ℂ) * Complex.I) = Complex.cosh (φ : ℂ)
      ∧ Complex.sin ((φ : ℂ) * Complex.I) = Complex.sinh (φ : ℂ) * Complex.I :=
  ⟨Complex.cos_mul_I _, Complex.sin_mul_I _⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE SWAP IS THE i² = −1 ↔ j² = +1 SIGN FLIP
-- ═══════════════════════════════════════════════════════════════════

/-- **The two cell units, and what Wick rotation reconciles.** The phase (Being) cell's
    unit squares to `−1` (`Complex.I_sq`); the Null cell's unit squares to `+1`
    (`NullCell.nj_sq`). Wick rotation is precisely the substitution that turns one into
    the other — imaginary time carries the real-time unit (square `+1`) to one squaring
    to `−1` (`imaginary_time_flips_sign`). The whole rotation is this one sign. -/
theorem cell_units_swap :
    Complex.I ^ 2 = -1 ∧ nmul nj nj = nOne :=
  ⟨Complex.I_sq, nj_sq⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE HEADLINE
-- ═══════════════════════════════════════════════════════════════════

/-- **Wick rotation, the Being ↔ Null swap (headline).** One operation — imaginary time
    `b ↦ i·b` — carries the Lorentzian (Null-cell) world to the Euclidean (phase-cell)
    world, on every face at once:

    1. **metric** — the interval `a² − b²` (Null, real time) becomes `a² + b²` (Being,
       imaginary time): the only thing that moves is the sign of the time-squared term;
    2. **dynamics** — the boost `cosh` becomes the rotor `cos` (`cos(iφ) = cosh φ`);
    3. **unit** — it is the `i² = −1` ↔ `j² = +1` sign flip, the whole of Non-Being.

    Hence the corpus's naive Madelung/HJ picture (the phase-cell rotor on a smooth
    Euclidean stage, time as a phase advance) is the **Wick-rotated shadow** of the
    one-Null-cell Lorentzian spacetime: un-rotate one spinning disk into a lightcone
    funnel and the smooth stage acquires its single arrow of time and its causal cone. -/
theorem wick_rotation :
    (∀ a b : ℝ, (a : ℂ) ^ 2 - ((b : ℂ) * Complex.I) ^ 2 = (a : ℂ) ^ 2 + (b : ℂ) ^ 2)
      ∧ (∀ φ : ℝ, Complex.cos ((φ : ℂ) * Complex.I) = Complex.cosh (φ : ℂ))
      ∧ (Complex.I ^ 2 = -1 ∧ nmul nj nj = nOne) :=
  ⟨wick_metric, fun φ => Complex.cos_mul_I (φ : ℂ), cell_units_swap⟩

end CrossUniverse

-- ═══════════════════════════════════════════════════════════════════
-- THE TOWER CHORD — WICK SWAP IS THE STEP-KIND FLIP
-- ═══════════════════════════════════════════════════════════════════

/-! The bulk tower (`BulkTower`) routes each level's state into either the
    next boundary (`stepB`, the Being side, square `−1`) or the next drag
    (`stepU`, the Null side, square `+1`). These two step kinds ARE the two
    cells the Wick rotation relates: Wick rotation on the cell unit
    (`CrossUniverse.cell_units_swap`) corresponds to swapping the tower's step
    kind. Here that swap is realised as `wickSwap`, and shown to exchange the
    Being / Null transition counters and to be an involution — the `i²=−1`
    applied twice returning to `+1`, on the tower's own counting measure. -/

namespace BulkTower

/-- **Wick swap on the tower: flip every step kind.** Each `stepB` (Being
    routing) becomes `stepU` (Null routing) and vice versa, carrying the same
    `ℝ → ℝ` data. This is the Being ↔ Null cell swap (`cell_units_swap`)
    acting on the tower's transition structure. -/
def wickSwap : BulkTower → BulkTower
  | .base e => .base e
  | .stepB prev U_outer => .stepU U_outer prev.wickSwap
  | .stepU B_outer prev => .stepB prev.wickSwap B_outer

/-- **The swap exchanges the Being / Null counters.** Wick-swapping the tower
    turns every Being-routed (`stepB`) transition into a Null-routed (`stepU`)
    one, so the `stepU` count after swapping is the `stepB` count before —
    the counting-measure shadow of the unit sign flip `i²=−1 ↔ j²=+1`. -/
theorem wickSwap_stepU_count (t : BulkTower) :
    t.wickSwap.stepU_count = t.stepB_count := by
  induction t with
  | base e => rfl
  | stepB prev U_outer ih => simp [wickSwap, stepU_count, stepB_count, ih]
  | stepU B_outer prev ih => simp [wickSwap, stepU_count, stepB_count, ih]

/-- Dually, the swap sends the Null count to the prior Being count's partner. -/
theorem wickSwap_stepB_count (t : BulkTower) :
    t.wickSwap.stepB_count = t.stepU_count := by
  induction t with
  | base e => rfl
  | stepB prev U_outer ih => simp [wickSwap, stepU_count, stepB_count, ih]
  | stepU B_outer prev ih => simp [wickSwap, stepU_count, stepB_count, ih]

/-- **The swap preserves depth.** Flipping step kinds never adds or drops a
    transition, so the tower's height is Wick-invariant — only the sign
    (Being vs Null) of each step moves, exactly as the metric swap moves only
    the time-squared sign (`wick_metric`). -/
theorem wickSwap_depth (t : BulkTower) :
    t.wickSwap.depth = t.depth := by
  induction t with
  | base e => rfl
  | stepB prev U_outer ih => simp [wickSwap, depth, ih]
  | stepU B_outer prev ih => simp [wickSwap, depth, ih]

/-- **Wick swap is an involution.** Applying the swap twice returns the tower
    unchanged: the rotor → boost → rotor round trip, the `i²=−1`-twice = `+1`
    return at the level of the tower's structure. Un-Wick-rotating a swapped
    tower restores it exactly. -/
theorem wickSwap_wickSwap (t : BulkTower) :
    t.wickSwap.wickSwap = t := by
  induction t with
  | base e => rfl
  | stepB prev U_outer ih => simp [wickSwap, ih]
  | stepU B_outer prev ih => simp [wickSwap, ih]

/-- **The tower chord (headline).** The Wick rotation's Being ↔ Null cell swap
    (`CrossUniverse.cell_units_swap`, `i²=−1 ↔ j²=+1`) is realised on the bulk tower as
    `wickSwap`: it (1) exchanges the Being-routing and Null-routing transition
    counters, (2) preserves depth (only the sign of each step moves), and (3)
    is an involution (the round trip returns). The tower's two step kinds ARE
    the two cells, and Wick rotation swaps them.

    Native: the algebra of `wickSwap` on `BulkTower`. Cited-bridge: the
    identification of `stepB`/`stepU` with the Being/Null cells and of this
    swap with physical Wick rotation — the same interpreted reading as the
    rest of this file. -/
theorem wick_chord (t : BulkTower) :
    t.wickSwap.stepU_count = t.stepB_count
      ∧ t.wickSwap.stepB_count = t.stepU_count
      ∧ t.wickSwap.depth = t.depth
      ∧ t.wickSwap.wickSwap = t :=
  ⟨t.wickSwap_stepU_count, t.wickSwap_stepB_count,
   t.wickSwap_depth, t.wickSwap_wickSwap⟩

end BulkTower

end Sanguine

end
