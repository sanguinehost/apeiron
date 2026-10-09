import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.Carrier
import Apeiron.Physics.Carrier.Derived.Evolve
import Apeiron.Physics.CarrierDissipationAnyRing

/-!
# Diffusive relaxation of the carrier density

@name Carrier Diffusive Relaxation
@cite kocik-1999
@grade proved

A finite one-dimensional carrier line is relaxed toward a constant density by a shared-face
diffusive flux. Cell amount is read from `Carrier.carrierDensity` through `carrierState`; no
second material field is introduced. Face `i` sits between cells `i - 1` and `i`, so face `0` is
the left ghost face and face `n` is the right ghost face. Its flux is proportional to the
negative forward density difference, `-ν * (rho i - rho (i-1))`. The conservative update removes
the outgoing-minus-incoming flux, so local continuity holds by construction and the interior
faces cancel in the total. The explicit closed-boundary convention is `diffusiveFlux ν rho 0 = 0`
(the left ghost is built into the definition) together with the right hypothesis
`diffusiveFlux ν rho n = 0`, equivalently `slope rho (n-1) = 0` when `ν` is nonzero.

The exact algebraic stencil, the shared down-gradient flux, the finite update and total-mass
conservation live in `Apeiron.Physics.CarrierDissipationAnyRing`, over a module `M` over a
commutative ring `R`. This file is the ordered-real instance: it consumes `slope`, `prevSlope`,
`smoothSlope`, `diffusiveFlux` and `diffusiveUpdate` from that file and adds the claims that need
`ℝ`. The finite Dirichlet energy is the sum of squared adjacent differences over the edges that
join two logical cells, `dirichletEnergy rho m = ∑ i < m, (rho (i+1) - rho i)²`. The ghost edges
`-1` and `m` are not summed; they enter only through the two zero-flux boundary hypotheses. The
coefficient `ν` is an explicit gradient-flow time-step parameter, not a derived physical
viscosity, and no number here is fitted to a measurement. The stable range proved below is
`0 ≤ ν ≤ 1/2`; the bound follows from the convex weights of the update and is not forced onto the
indexing. The negative control exhibits an overlarge step, `ν = 1`, that strictly raises the
energy of a nonconstant finite witness.

On three cells with the right ghost face closed, the half step `ν = 1/2` averages every cell with
its neighbor, so the two interior differences are the old differences halved and swapped.
`dirichletEnergy_halfStep_three` is then the exact identity `E' = (1/4) E`, and
`dirichletEnergy_halfStep_three_lt` is strict whenever `E > 0`, that is, whenever the three cells
are not all equal. This is sharper than the convexity bound `dirichletEnergy_nonincreasing`: the
factor `1/4` is a property of the field. A general commutative ring has no division by two, and
the order-free kernel states the same face through the `exactHalfFlux` witness `DoublesTo`, which
exists only when the forward difference is even; the integer exact-even execution is therefore
exact only on even differences, while the real identity needs no evenness hypothesis.

## Supported regime, assumptions, error and work bounds

The regime is a one-dimensional, unit-spacing, closed carrier line with one shared flux per face.
The step is first-order in time and reads one forward difference at unit spacing. One update over
`n` cells takes work linear in `n` with constant work per cell, and the proofs add no runtime
operation. There is no body work, no external forcing, no stored circulation, and no physically
derived damping coefficient. This is a constitutive relaxation experiment, not a general fluid
solver: it has no stored height, velocity or pressure, no viscosity model and no pressure solve.

## Root

`carrierState` is `Carrier.carrierDensity` on the line, so the density authority stays the
trichotomy norm form. The stencil and the update are the generic ones from
`Apeiron.Physics.CarrierDissipationAnyRing`; this file proves that the density is nonnegative,
that the update preserves nonnegativity on the stable range, and that the Dirichlet energy does
not increase. `dirichletEnergy_nonnegative` is a sum of squares, and
`dirichletEnergy_nonincreasing` rests on the convexity of the square and the two ghost sums.
`dirichletEnergy_halfStep_three` is the sharp three-cell identity on the same update, composed
from `diffusiveUpdate_half_three` rather than a second update.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- The carrier density readout on the finite one-dimensional line.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →  [takes the left and gives the right; between statements, if the left holds so does the right]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    The authoritative state is `Carrier.carrierDensity`; this is that readout, not an independent
    field, and every theorem below is a statement about it once specialized. -/
noncomputable def carrierState (σ : ℝ) (R : PhaseField) (u : Cell) : ℕ → ℝ :=
  carrierDensity σ R u

/-- The finite Dirichlet energy of the density array.

    (rho : ℕ → ℝ)                   [the density array]
    (m : ℕ)                         [the number of interior edges]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ∈                               [is one of the things in]

    The sum runs over the edges `0` to `m-1`, each joining two logical cells; the ghost edges are
    excluded. Every term is a squared adjacent difference, so the value is at or above zero. -/
def dirichletEnergy (rho : ℕ → ℝ) (m : ℕ) : ℝ :=
  ∑ i ∈ Finset.range m, slope rho i ^ 2

/-- The carrier state is the carrier density readout, by definition.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (i : ℕ)                         [the counting cell position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    This is the tie that keeps the arbitrary array in the general lemmas from being read as an
    independent material field. -/
theorem carrierState_eq (σ : ℝ) (R : PhaseField) (u : Cell) (i : ℕ) :
    carrierState σ R u i = carrierDensity σ R u i := rfl

/-- The carrier state is never negative when the phase direction has size one.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    (i : ℕ)                         [the counting cell position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    A square read from `Carrier.density_nonnegative`, so the wet/dry seam is amplitude zero. -/
theorem carrierState_nonnegative (σ : ℝ) (R : PhaseField) (u : Cell)
    (hu : normForm σ u = 1) (i : ℕ) : 0 ≤ carrierState σ R u i := by
  rw [carrierState, carrierDensity]
  exact density_nonnegative σ R u (i : ℝ) 0 hu

/-- A pointwise nonnegative density stays nonnegative under a stable diffusive step.

    (ν : ℝ)                         [the explicit gradient-flow time-step parameter]
    (hν0 : 0 ≤ ν)                   [the time-step parameter is at or above zero]
    (hν : ν ≤ 1/2)                  [the time-step parameter is at most one half]
    (rho : ℕ → ℝ)                   [the density array]
    (hrho : ∀ i, 0 ≤ rho i)         [every cell amount is at or above zero]
    (i : ℕ)                         [the counting cell position]
    0 ≤ diffusiveUpdate ν rho i     [the updated amount is at or above zero]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ∀                               [for every single one]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]

    On `0 ≤ ν ≤ 1/2` the weights `(1 - ν)`, `ν`, and `1 - 2ν` are nonnegative and sum to one,
    so `diffusiveUpdate_zero` and `diffusiveUpdate_succ` express the update as a convex
    combination of nonnegative cell amounts. -/
theorem diffusiveUpdate_nonnegative {ν : ℝ} (hν0 : 0 ≤ ν) (hν : ν ≤ 1 / 2)
    {rho : ℕ → ℝ} (hrho : ∀ i, 0 ≤ rho i) (i : ℕ) :
    0 ≤ diffusiveUpdate ν rho i := by
  have h1ν : 0 ≤ 1 - ν := by linarith
  have h12ν : 0 ≤ 1 - 2 * ν := by linarith
  cases i with
  | zero =>
      rw [diffusiveUpdate_zero]
      simp only [smul_eq_mul]
      exact add_nonneg (mul_nonneg h1ν (hrho 0)) (mul_nonneg hν0 (hrho 1))
  | succ k =>
      rw [diffusiveUpdate_succ]
      simp only [smul_eq_mul]
      exact add_nonneg
        (add_nonneg (mul_nonneg hν0 (hrho k))
          (mul_nonneg h12ν (hrho (k + 1))))
        (mul_nonneg hν0 (hrho (k + 2)))

/-- The finite Dirichlet energy is never negative.

    (rho : ℕ → ℝ)                   [the density array]
    (m : ℕ)                         [the number of interior edges]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ≤                               [is smaller than or equal to]

    Every summand is a square, so the energy is at or above zero. -/
theorem dirichletEnergy_nonnegative (rho : ℕ → ℝ) (m : ℕ) :
    0 ≤ dirichletEnergy rho m := by
  rw [dirichletEnergy]
  exact Finset.sum_nonneg fun i _ => sq_nonneg _

/-- A convex combination of three reals has square at most the weighted square sum.

    (a b c x y z : ℝ)               [three nonnegative weights and three real values]
    (ha : 0 ≤ a)                    [the first weight is at or above zero]
    (hb : 0 ≤ b)                    [the second weight is at or above zero]
    (hc : 0 ≤ c)                    [the third weight is at or above zero]
    (hsum : a + b + c = 1)          [the three weights add to one]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]

    The identity behind it is a sum of three nonnegative cross terms, `ab(x-y)² + ac(x-z)² +
    bc(y-z)²`, which is the elementary square bound the energy proof uses. -/
theorem sq_weighted_le (a b c x y z : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hsum : a + b + c = 1) :
    (a * x + b * y + c * z) ^ 2 ≤ a * x ^ 2 + b * y ^ 2 + c * z ^ 2 := by
  have hc' : c = 1 - a - b := by linarith
  have h1 : 0 ≤ a * b * (x - y) ^ 2 := by positivity
  have h2 : 0 ≤ a * c * (x - z) ^ 2 := by positivity
  have h3 : 0 ≤ b * c * (y - z) ^ 2 := by positivity
  have key : a * x ^ 2 + b * y ^ 2 + c * z ^ 2 - (a * x + b * y + c * z) ^ 2
      = a * b * (x - y) ^ 2 + a * c * (x - z) ^ 2 + b * c * (y - z) ^ 2 := by
    rw [hc']
    ring
  linarith

/-- The backward-neighbor sum is the shortened energy sum.

    (d : ℕ → ℝ)                     [a real difference array]
    (m : ℕ)                         [the number of interior edges]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ∈                               [is one of the things in]

    The left ghost neighbor is zero and every later backward neighbor is the previous difference,
    so this is the energy sum with its last term removed. -/
theorem sum_prevSlope_sq (d : ℕ → ℝ) (m : ℕ) :
    (∑ i ∈ Finset.range m, prevSlope d i ^ 2)
      = ∑ i ∈ Finset.range (m - 1), d i ^ 2 := by
  cases m with
  | zero => simp [prevSlope]
  | succ m' =>
      rw [Finset.sum_range_succ']
      simp [prevSlope]

/-- The forward-shifted sum is the energy sum less its first term, with the closed right end.

    (d : ℕ → ℝ)                     [a real difference array]
    (m : ℕ)                         [the number of interior edges]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ∈                               [is one of the things in]

    The shift loses the first squared difference and gains the ghost edge `m`, which the closed
    boundary sets to zero. -/
theorem sum_slope_succ_sq (d : ℕ → ℝ) (m : ℕ) :
    (∑ i ∈ Finset.range m, d (i + 1) ^ 2)
      = (∑ i ∈ Finset.range m, d i ^ 2) - d 0 ^ 2 + d m ^ 2 := by
  induction m with
  | zero => simp
  | succ m ih =>
      rw [Finset.sum_range_succ (fun i => d (i + 1) ^ 2),
        Finset.sum_range_succ (fun i => d i ^ 2), ih]
      ring

/-- The smoothed difference array does not raise the finite Dirichlet energy.

    (ν : ℝ)                         [the explicit gradient-flow time-step parameter]
    (hν0 : 0 ≤ ν)                   [the time-step parameter is at or above zero]
    (hν : ν ≤ 1/2)                  [the time-step parameter is at most one half]
    (d : ℕ → ℝ)                     [a real difference array]
    (m : ℕ)                         [the number of interior edges]
    (hm : d m = 0)                  [the right ghost edge is zero]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ≤                               [is smaller than or equal to]
    ∈                               [is one of the things in]

    On the stable range the update is a convex combination of each difference and its two
    neighbors, so the square is at most the weighted square sum; summing, the two shifts are
    bounded by the same energy and the left ghost term is nonnegative. -/
theorem smoothSlope_energy_le (ν : ℝ) (hν0 : 0 ≤ ν) (hν : ν ≤ 1/2)
    (d : ℕ → ℝ) (m : ℕ) (hm : d m = 0) :
    (∑ i ∈ Finset.range m, smoothSlope ν d i ^ 2)
      ≤ ∑ i ∈ Finset.range m, d i ^ 2 := by
  have hw : (1 - 2 * ν) + ν + ν = 1 := by ring
  have hconv : ∀ i, smoothSlope ν d i ^ 2
      ≤ (1 - 2 * ν) * d i ^ 2 + ν * prevSlope d i ^ 2 + ν * d (i + 1) ^ 2 := by
    intro i
    simp only [smoothSlope, smul_eq_mul]
    exact sq_weighted_le (1 - 2 * ν) ν ν (d i) (prevSlope d i) (d (i + 1))
      (by linarith) hν0 hν0 hw
  have hP : (∑ i ∈ Finset.range m, prevSlope d i ^ 2)
      ≤ ∑ i ∈ Finset.range m, d i ^ 2 := by
    rw [sum_prevSlope_sq]
    exact Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.range_subset_range.mpr (Nat.sub_le m 1)) (fun i _ _ => sq_nonneg _)
  have hN : (∑ i ∈ Finset.range m, d (i + 1) ^ 2)
      = (∑ i ∈ Finset.range m, d i ^ 2) - d 0 ^ 2 := by
    rw [sum_slope_succ_sq, hm]
    ring
  have hE : (0 : ℝ) ≤ ∑ i ∈ Finset.range m, d i ^ 2 :=
    Finset.sum_nonneg fun i _ => sq_nonneg _
  have hcoef : (0 : ℝ) ≤ 1 - 2 * ν := by linarith
  calc
    (∑ i ∈ Finset.range m, smoothSlope ν d i ^ 2)
      ≤ ∑ i ∈ Finset.range m, ((1 - 2 * ν) * d i ^ 2 + ν * prevSlope d i ^ 2
          + ν * d (i + 1) ^ 2) := Finset.sum_le_sum fun i _ => hconv i
    _ = (1 - 2 * ν) * (∑ i ∈ Finset.range m, d i ^ 2)
          + ν * (∑ i ∈ Finset.range m, prevSlope d i ^ 2)
          + ν * (∑ i ∈ Finset.range m, d (i + 1) ^ 2) := by
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum,
          Finset.mul_sum]
    _ ≤ (1 - 2 * ν) * (∑ i ∈ Finset.range m, d i ^ 2)
          + ν * (∑ i ∈ Finset.range m, d i ^ 2)
          + ν * ((∑ i ∈ Finset.range m, d i ^ 2) - d 0 ^ 2) := by
        nlinarith [hP, hN, hE, hcoef, hν0]
    _ = (∑ i ∈ Finset.range m, d i ^ 2) - ν * d 0 ^ 2 := by ring
    _ ≤ ∑ i ∈ Finset.range m, d i ^ 2 := by
        nlinarith [mul_nonneg hν0 (sq_nonneg (d 0))]

/-- The conservative diffusive update does not raise the finite Dirichlet energy.

    (ν : ℝ)                         [the explicit gradient-flow time-step parameter]
    (hν0 : 0 ≤ ν)                   [the time-step parameter is at or above zero]
    (hν : ν ≤ 1/2)                  [the time-step parameter is at most one half]
    (rho : ℕ → ℝ)                   [the density array]
    (m : ℕ)                         [the number of interior edges]
    (hm : slope rho m = 0)          [the right ghost edge carries no flux]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ≤                               [is smaller than or equal to]
    ∈                               [is one of the things in]

    The updated differences are the smoothed differences, so the abstract energy bound applies at
    the closed right end. -/
theorem dirichletEnergy_nonincreasing (ν : ℝ) (hν0 : 0 ≤ ν) (hν : ν ≤ 1/2)
    (rho : ℕ → ℝ) (m : ℕ) (hm : slope rho m = 0) :
    dirichletEnergy (diffusiveUpdate ν rho) m ≤ dirichletEnergy rho m := by
  rw [dirichletEnergy, dirichletEnergy]
  simp_rw [slope_diffusiveUpdate]
  exact smoothSlope_energy_le ν hν0 hν (slope rho) m hm

/-- The intended carrier instantiation inherits the energy bound.

    (ν : ℝ)                         [the explicit gradient-flow time-step parameter]
    (hν0 : 0 ≤ ν)                   [the time-step parameter is at or above zero]
    (hν : ν ≤ 1/2)                  [the time-step parameter is at most one half]
    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (m : ℕ)                         [the number of interior edges]
    (hm : slope (carrierState σ R u) m = 0)
                                     [the carrier's right ghost edge carries no flux]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ≤                               [is smaller than or equal to]
    ∈                               [is one of the things in]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    This is the general theorem specialized to `carrierState`, so the physical density readout is
    the state the bound is about. -/
theorem carrierState_energy_nonincreasing (ν : ℝ) (hν0 : 0 ≤ ν) (hν : ν ≤ 1/2)
    (σ : ℝ) (R : PhaseField) (u : Cell) (m : ℕ)
    (hm : slope (carrierState σ R u) m = 0) :
    dirichletEnergy (diffusiveUpdate ν (carrierState σ R u)) m
      ≤ dirichletEnergy (carrierState σ R u) m :=
  dirichletEnergy_nonincreasing ν hν0 hν (carrierState σ R u) m hm

/-- The closed three-cell half step averages every cell with its neighbor.

    (rho : ℕ → ℝ)                   [the density array]
    (hRight : diffusiveFlux (1 / 2 : ℝ) rho 3 = 0)
                                     [the right ghost face carries no flux]
    diffusiveUpdate (1 / 2 : ℝ) rho 0 = (rho 0 + rho 1) / 2
                                     [the first cell averages with cell 1]
    ∧ diffusiveUpdate (1 / 2 : ℝ) rho 1 = (rho 0 + rho 2) / 2
                                     [the middle cell averages its two neighbors]
    ∧ diffusiveUpdate (1 / 2 : ℝ) rho 2 = (rho 1 + rho 2) / 2
                                     [the last cell averages with cell 1]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    `diffusiveFlux`                 [the shared face flux: the negative forward difference]
    ∧                               [both are true]

    With `ν = 1/2` the interior weight `1 - 2ν` vanishes, so the update is a two-point average
    everywhere. The closed right end is the hypothesis at face 3, the face the update of cell 2
    reads; without it cell 2 keeps its two-sided stencil. -/
theorem diffusiveUpdate_half_three (rho : ℕ → ℝ)
    (hRight : diffusiveFlux (1 / 2 : ℝ) rho 3 = 0) :
    diffusiveUpdate (1 / 2 : ℝ) rho 0 = (rho 0 + rho 1) / 2 ∧
    diffusiveUpdate (1 / 2 : ℝ) rho 1 = (rho 0 + rho 2) / 2 ∧
    diffusiveUpdate (1 / 2 : ℝ) rho 2 = (rho 1 + rho 2) / 2 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [diffusiveUpdate_zero]
    simp only [smul_eq_mul]
    ring
  · rw [diffusiveUpdate_succ]
    simp only [smul_eq_mul]
    ring
  · rw [diffusiveUpdate, hRight]
    simp only [diffusiveFlux, smul_eq_mul]
    ring

/-- One exact half step quarters the three-cell Dirichlet energy.

    (rho : ℕ → ℝ)                   [the density array]
    (hRight : diffusiveFlux (1 / 2 : ℝ) rho 3 = 0)
                                     [the right ghost face carries no flux]
    dirichletEnergy (diffusiveUpdate (1 / 2 : ℝ) rho) 2 = (1 / 4 : ℝ) * dirichletEnergy rho 2
                                     [the updated energy is a quarter of the old]
    `dirichletEnergy`               [the finite Dirichlet energy: the squared adjacent differences]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]

    The update halves each interior difference and swaps the two, so the squared sum is the old
    squared sum times `1/4`. This is an exact identity, not the convexity bound
    `dirichletEnergy_nonincreasing`. Over `ℝ` the factor `1/4` is available for every value; the
    order-free kernel states the same face through the `exactHalfFlux` witness, which a ring
    without two invertible, such as an integer or a machine word, supplies only when the
    difference is even. -/
theorem dirichletEnergy_halfStep_three (rho : ℕ → ℝ)
    (hRight : diffusiveFlux (1 / 2 : ℝ) rho 3 = 0) :
    dirichletEnergy (diffusiveUpdate (1 / 2 : ℝ) rho) 2 = (1 / 4 : ℝ) * dirichletEnergy rho 2 := by
  obtain ⟨h0, h1, h2⟩ := diffusiveUpdate_half_three rho hRight
  rw [dirichletEnergy, dirichletEnergy]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, slope, h0, h1, h2]
  ring

/-- A positive three-cell energy strictly decreases under the exact half step.

    (rho : ℕ → ℝ)                   [the density array]
    (hRight : diffusiveFlux (1 / 2 : ℝ) rho 3 = 0)
                                     [the right ghost face carries no flux]
    (hpos : 0 < dirichletEnergy rho 2)
                                     [the old adjacent-difference energy is positive]
    dirichletEnergy (diffusiveUpdate (1 / 2 : ℝ) rho) 2 < dirichletEnergy rho 2
                                     [the updated energy is strictly below the old energy]
    `dirichletEnergy`               [the finite Dirichlet energy: the squared adjacent differences]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    <                               [is smaller than]

    A positive sum of two squares is a nonconstant three-cell state: it vanishes exactly when
    both adjacent differences do. The quarter identity then gives the strict inequality. -/
theorem dirichletEnergy_halfStep_three_lt (rho : ℕ → ℝ)
    (hRight : diffusiveFlux (1 / 2 : ℝ) rho 3 = 0)
    (hpos : 0 < dirichletEnergy rho 2) :
    dirichletEnergy (diffusiveUpdate (1 / 2 : ℝ) rho) 2 < dirichletEnergy rho 2 := by
  rw [dirichletEnergy_halfStep_three rho hRight]
  nlinarith

/-- The negative-control density witness, a nonconstant finite state.

    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →  [takes the left and gives the right; between statements, if the left holds so does the right]

    Cell `1` holds one and every other cell holds zero, so the state is nonconstant and its
    differences are `1`, `-1`, `0` over the three edges. -/
def controlState : ℕ → ℝ := fun i => if i = 1 then 1 else 0

/-- An overlarge step strictly raises the energy of the negative-control witness.

    `dirichletEnergy`               [the finite Dirichlet energy: the squared adjacent differences]
    `controlState`                  [the negative-control density witness]
    ℝ                               [the real numbers: every number on the number line]

    At `ν = 1`, outside the stable range, the witness `controlState` has energy `2` while its
    update has energy `8`, so the stable bound is not vacuous. This is the Lean shape the Rust
    negative control replays. -/
theorem overlarge_step_increases_energy :
    dirichletEnergy (diffusiveUpdate (1 : ℝ) controlState) 2 > dirichletEnergy controlState 2 := by
  norm_num [dirichletEnergy, slope, diffusiveUpdate, diffusiveFlux, controlState,
    Finset.sum_range_succ, Finset.sum_range_zero]

/-- The three-cell half-step hypotheses hold on the negative-control witness.

    `controlState`                  [the negative-control density witness]
    `diffusiveFlux`                 [the shared face flux: the negative forward difference]
    `dirichletEnergy`               [the finite Dirichlet energy: the squared adjacent differences]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    ℝ                               [the real numbers: every number on the number line]
    ∧                               [both are true]
    <                               [is smaller than]
    diffusiveFlux (1 / 2 : ℝ) controlState 3 = 0
                                     [the right ghost face carries no flux]
    ∧ 0 < dirichletEnergy controlState 2
                                     [the old adjacent-difference energy is positive]
    ∧ dirichletEnergy (diffusiveUpdate (1 / 2 : ℝ) controlState) 2
        < dirichletEnergy controlState 2
                                     [the updated energy is strictly below the old energy]

    The witness is one at cell `1` and zero elsewhere, so face `3` is closed, the old energy is
    `2`, and the half step lowers it. This is the non-vacuity witness for
    `dirichletEnergy_halfStep_three_lt`, not a second statement of the bound. -/
theorem halfStep_three_control :
    diffusiveFlux (1 / 2 : ℝ) controlState 3 = 0 ∧
    0 < dirichletEnergy controlState 2 ∧
    dirichletEnergy (diffusiveUpdate (1 / 2 : ℝ) controlState) 2
      < dirichletEnergy controlState 2 := by
  have hRight : diffusiveFlux (1 / 2 : ℝ) controlState 3 = 0 := by
    norm_num [diffusiveFlux, controlState, smul_eq_mul]
  have hpos : 0 < dirichletEnergy controlState 2 := by
    norm_num [dirichletEnergy, slope, controlState, Finset.sum_range_succ, Finset.sum_range_zero]
  exact ⟨hRight, hpos, dirichletEnergy_halfStep_three_lt controlState hRight hpos⟩

end Physics
end Apeiron
