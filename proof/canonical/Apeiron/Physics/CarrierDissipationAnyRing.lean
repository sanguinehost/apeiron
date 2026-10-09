import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.CarrierAnyRing

/-!
# Diffusive relaxation over any commutative ring

@name Carrier Diffusive Relaxation Over Any Commutative Ring
@cite kocik-1999
@grade proved

`Apeiron.Physics.CarrierDissipation` relaxes a finite carrier line with a shared-face diffusive
flux, but it states the stencil, the update and total-mass conservation for `ℝ` and consults the
order only later, for the Dirichlet energy. The stencil itself never reads an order: it is a
forward difference, a scalar multiple and a subtraction. This file carries that algebraic core to
a module `M` over a commutative ring `R`, so `ℝ` and a machine word are instances of one update
rather than two restatements. It is rooted after `Apeiron.Physics.CarrierAnyRing`, which installs
the ring-level density readout the real file samples.

`slope` is the forward difference `rho (i + 1) - rho i`. `diffusiveFlux` is the shared
down-gradient flux `-ν • prevSlope (slope rho) i`, with the left ghost face set to zero by
`prevSlope`. `diffusiveUpdate` subtracts the outgoing-minus-incoming flux, so
`diffusiveUpdate_continuity` and `diffusiveUpdate_total_mass` are consequences of the definition.
The scalar `ν : R` acts through the module structure and no order enters.

The ordered claims stay in `Apeiron.Physics.CarrierDissipation`: density nonnegativity, the stable
range `0 ≤ ν ≤ 1/2`, the Dirichlet energy bound and the overlarge-step control are instance
theorems for `ℝ`. They consume `smoothSlope` and `slope_diffusiveUpdate` from here and do not
redefine the update.

## Root

| from | here |
|---|---|
| `CarrierAnyRing.pointDensity` | `diffusiveFlux`, `diffusiveUpdate` |

The stencil is a forward difference, a scalar multiple and a subtraction over an arbitrary module
`M` over a commutative ring `R`, so it uses no `RingCell` operation directly. The definitions are
`slope`, `prevSlope`, `smoothSlope`, `diffusiveFlux` and `diffusiveUpdate`, and
`diffusiveFlux_eq`, `diffusiveUpdate_continuity` and `diffusiveUpdate_total_mass` follow from
them. The readout this update acts on is `CarrierAnyRing.pointDensity`. What does not transfer is
every ordered claim: density nonnegativity, the stable range `0 ≤ ν ≤ 1/2`, the Dirichlet energy
bound and the overlarge-step control are `ℝ` results in `Apeiron.Physics.CarrierDissipation`.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Physics

variable {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M]

/-- The forward difference of a module-valued array.

    (rho : ℕ → M)                   [the density array]
    (i : ℕ)                         [the counting position of the forward edge]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    Edge `i` joins cell `i` and cell `i+1`; this is the adjacent difference the real energy
    squares. The module `M` replaces `ℝ` and no order is used. -/
def slope (rho : ℕ → M) (i : ℕ) : M := rho (i + 1) - rho i

/-- The backward neighbor of a difference array, zero at the left ghost edge.

    (d : ℕ → M)                     [a difference array]
    (i : ℕ)                         [the counting position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    The value at `0` is the explicit left ghost edge, so the smoothing at the left boundary sees a
    zero neighbor. -/
def prevSlope (d : ℕ → M) (i : ℕ) : M := if i = 0 then 0 else d (i - 1)

/-- One diffusive smoothing of a difference array with closed ends.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (d : ℕ → M)                     [a difference array]
    (i : ℕ)                         [the counting position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    The weights `1 - 2ν`, `ν`, `ν` are the update's convex combination when `0 ≤ ν ≤ 1/2`; over a
    general module the combination is written with the scalar action. -/
def smoothSlope (ν : R) (d : ℕ → M) (i : ℕ) : M :=
  (1 - 2 * ν) • d i + ν • prevSlope d i + ν • d (i + 1)

/-- The diffusive flux on the shared face between two adjacent cells.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → M)                   [the density array]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    Face `i` joins cell `i - 1` and cell `i` and carries `-ν • (rho i - rho (i-1))`, the negative
    forward density difference. Face `0` is the left ghost face and carries exactly zero. -/
def diffusiveFlux (ν : R) (rho : ℕ → M) : ℕ → M
  | 0 => 0
  | (i + 1) => -ν • (rho (i + 1) - rho i)

/-- The conservative diffusive update of the density array.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → M)                   [the density array]
    (i : ℕ)                         [the counting cell position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    Cell `i` loses the outgoing face `i+1` and gains the incoming face `i`, which is exactly old
    density minus outgoing-minus-incoming flux. -/
def diffusiveUpdate (ν : R) (rho : ℕ → M) (i : ℕ) : M :=
  rho i - (diffusiveFlux ν rho (i + 1) - diffusiveFlux ν rho i)

/-- The face flux is the negative forward difference of the density.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → M)                   [the density array]
    (i : ℕ)                         [the counting face position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    •                               [stretch by, a number times a point]

    The left ghost face is zero, and every later face reads the backward difference, so this is
    the single-valued shared flux with the explicit ghost convention. -/
theorem diffusiveFlux_eq (ν : R) (rho : ℕ → M) (i : ℕ) :
    diffusiveFlux ν rho i = -ν • prevSlope (slope rho) i := by
  cases i with
  | zero => simp [diffusiveFlux, prevSlope]
  | succ k => simp [diffusiveFlux, prevSlope, slope]

/-- One conservative update gives the same smoothing to the adjacent differences.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → M)                   [the density array]
    (i : ℕ)                         [the counting edge position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    The difference of the updated density is the convex smoothing of the old differences, so the
    real energy can be bounded without a second derivative. -/
theorem slope_diffusiveUpdate (ν : R) (rho : ℕ → M) (i : ℕ) :
    slope (diffusiveUpdate ν rho) i = smoothSlope ν (slope rho) i := by
  cases i with
  | zero => simp [slope, diffusiveUpdate, diffusiveFlux, smoothSlope, prevSlope]; module
  | succ k => simp [slope, diffusiveUpdate, diffusiveFlux, smoothSlope, prevSlope]; module

/-- The constructed update satisfies discrete continuity at every cell.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → M)                   [the density array]
    (i : ℕ)                         [the counting cell position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    This is `rho' - rho + div J = 0`, obtained by unfolding the definition, so continuity is a
    consequence of the update rather than an added assumption. -/
theorem diffusiveUpdate_continuity (ν : R) (rho : ℕ → M) (i : ℕ) :
    diffusiveUpdate ν rho i - rho i
      + (diffusiveFlux ν rho (i + 1) - diffusiveFlux ν rho i) = 0 := by
  rw [diffusiveUpdate]
  abel

/-- The update at the left boundary is the explicit two-cell formula.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → M)                   [the density array]
    diffusiveUpdate ν rho 0
      = (1 - ν) • rho 0 + ν • rho 1 [the left cell keeps `1 - ν` and takes `ν` of its neighbor]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    •                               [stretch by, a number times a point]

    The left ghost face is exactly zero, so cell `0` has no left inflow: this is the actual
    boundary formula the update computes, not a separate assumption. -/
theorem diffusiveUpdate_zero (ν : R) (rho : ℕ → M) :
    diffusiveUpdate ν rho 0 = (1 - ν) • rho 0 + ν • rho 1 := by
  simp [diffusiveUpdate, diffusiveFlux]
  module

/-- The interior update is the explicit three-point stencil.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → M)                   [the density array]
    (i : ℕ)                         [the counting cell position]
    diffusiveUpdate ν rho (i + 1)
      = ν • rho i + (1 - 2 * ν) • rho (i + 1) + ν • rho (i + 2)
                                    [the interior cell keeps `1 - 2ν` and takes `ν` from each
                                     side]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    •                               [stretch by, a number times a point]

    The two faces bordering cell `i + 1` are read from `diffusiveFlux`, so this is the actual
    index formula the update computes. -/
theorem diffusiveUpdate_succ (ν : R) (rho : ℕ → M) (i : ℕ) :
    diffusiveUpdate ν rho (i + 1)
      = ν • rho i + (1 - 2 * ν) • rho (i + 1) + ν • rho (i + 2) := by
  simp [diffusiveUpdate, diffusiveFlux]
  module

/-- A closed finite line conserves total mass in one update.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → M)                   [the density array]
    (n : ℕ)                         [the number of cells]
    (hRight : diffusiveFlux ν rho n = 0)
                                     [the right ghost face carries zero flux]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ∈                               [is one of the things in]

    The interior faces cancel in the telescoping sum and the two ghost faces are zero, so old and
    new totals are equal. This is `Finset.sum_range_sub`, not a second update. -/
theorem diffusiveUpdate_total_mass (ν : R) (rho : ℕ → M) (n : ℕ)
    (hRight : diffusiveFlux ν rho n = 0) :
    (∑ i ∈ Finset.range n, diffusiveUpdate ν rho i)
      = ∑ i ∈ Finset.range n, rho i := by
  simp_rw [diffusiveUpdate]
  rw [Finset.sum_sub_distrib]
  have htel : (∑ i ∈ Finset.range n,
      (diffusiveFlux ν rho (i + 1) - diffusiveFlux ν rho i))
      = diffusiveFlux ν rho n - diffusiveFlux ν rho 0 :=
    Finset.sum_range_sub (diffusiveFlux ν rho) n
  rw [htel, hRight]
  simp [diffusiveFlux]

end Physics
end Apeiron
