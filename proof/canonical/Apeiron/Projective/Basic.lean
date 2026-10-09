import Apeiron.Trichotomy.NullBasis
import Apeiron.Projective.AnyRing
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic

/-!
# The projective line over the algebra

@name The Projective Line
@cite kisil-2010
@cite kisil-2007
@cite yaglom-1979
@grade proved

A point of the projective line is a pair `[b : u]` counted up to a common factor. Reading it as
one number means dividing, and dividing needs `u` to be invertible. Over the reals that is
`u ≠ 0`. **Over this algebra it is not.**

At `σ ≥ 0` there are points that are not zero and still cannot be divided by, and
`isUnit_iff_normForm_ne_zero` says exactly which: a point is invertible when its size is not
zero, and `normForm_zero_iff_channel_zero` says a size of zero means one of the two channels reads
zero. So the locus where the projective coordinate fails is the null cone, and the cone is two
lines rather than a point.

That single locus is what this corpus keeps meeting under different names:

- the degeneracy of the projective coordinate, here
- the light cone of the Minkowski half-plane at `σ = +1`, which is Kisil's hyperbolic case and
  the reason his conformal compactification needs a time-arrow condition
- the crossing `../song/src/prover/cone/cell.slang` reports as unsatisfiable, whose C4 says it is
  a different set from the cone and proves both directions

## What is Kisil's and what is this corpus's

The geometry is his. `ℝ[x]/(x² − σ)` for the three values, the `SL(2,ℝ)` action by Möbius
transformations, the invariant curves he calls cycles, and the reading of a point as projective:
all of that is the *Erlangen Program at Large*, and the corpus should have been citing it from
the start rather than arriving at it.

What this corpus adds is that the family is **complete**. `classification` proves every
two-dimensional unital commutative associative real algebra is one of the three, machine-checked,
so the three cases are not three worked examples but the whole of the possibility. Kisil's
programme states the three and works them; it does not prove there is no fourth.
-/

namespace Apeiron
namespace Projective

open Trichotomy

/-- A point of the projective line: a pair whose second slot can be divided by.

    (σ : ℝ)                          [sigma, a real number]
    ProjPair σ                       [a point of the projective line over that algebra]
    b u : Cell                       [the two slots]
    u_unit : normForm σ u ≠ 0        [the second slot lies off the cone]

    So: the condition is not that `u` is non-zero. At `σ ≥ 0` there are non-zero points on the
    cone that nothing divides, and `isUnit_iff_normForm_ne_zero` says being off the cone is
    exactly being invertible. -/
structure ProjPair (σ : ℝ) where
  /-- The first slot. -/
  b : Cell
  /-- The second slot, which must be divisible by. -/
  u : Cell
  /-- It lies off the cone, which is what makes it divisible by. -/
  u_unit : normForm σ u ≠ 0

/-- The second slot of a projective point can be divided by.

    {σ : ℝ} (p : ProjPair σ)              [a point of the projective line]
    ∃ w : Cell, mul σ p.u w = one         [something multiplies with its second slot to give one]

    So: the structure's own field, read through `isUnit_iff_normForm_ne_zero`. -/
theorem u_invertible {σ : ℝ} (p : ProjPair σ) : ∃ w : Cell, mul σ p.u w = one :=
  (isUnit_iff_normForm_ne_zero σ p.u).mpr p.u_unit

/-- The real projective pair is the generic one.

    {σ : ℝ} (p : ProjPair σ)      [a point of the projective line]
    : Generic.RingProjPair σ      [the same point with its size's unit made explicit]

    So: `ProjPair`'s field is `normForm σ u ≠ 0`, and over the reals that is `IsUnit (normForm σ
    u)`, which is the generic structure's field. This is the bridge from the real projective line
    to `Projective/AnyRing.lean`, and it is definitional in the two slots.

    The two structures cannot be one declaration. `ProjPair`'s third field is `normForm σ u ≠ 0`;
    `Generic.RingProjPair`'s third field is `IsUnit (ringNormForm σ u)`. Over `ℝ`,
    `isUnit_iff_ne_zero` makes the two propositions equivalent, but they are not definitionally
    equal, and a structure alias needs its field types to match definitionally. So this stays a
    function on the slots rather than an `abbrev`. -/
def ProjPair.toGeneric {σ : ℝ} (p : ProjPair σ) : Generic.RingProjPair σ where
  b := p.b
  u := p.u
  u_unit := isUnit_iff_ne_zero.mpr p.u_unit

/-- Neither channel of the second slot reads zero.

    {σ : ℝ} (hσ : 0 ≤ σ) (p : ProjPair σ)      [sigma at zero or above, and a point]
    upper σ p.u ≠ 0 ∧ lower σ p.u ≠ 0          [both channel readings are non-zero]

    So: the cone is two lines, one per channel (`normForm_zero_iff_channel_zero`), so lying off it
    means clearing both. This is what makes the two-channel reading of a projective point total
    where it is defined at all. -/
theorem channels_ne_zero {σ : ℝ} (hσ : 0 ≤ σ) (p : ProjPair σ) :
    upper σ p.u ≠ 0 ∧ lower σ p.u ≠ 0 := by
  constructor <;> intro h <;>
    exact p.u_unit ((normForm_zero_iff_channel_zero hσ p.u).mpr (by tauto))

-- ═══════════════════════════════════════════════════════════════════
-- READING THE PAIR AS ONE NUMBER
-- ═══════════════════════════════════════════════════════════════════

/-- Multiply a point by a scalar, slot by slot.

    (c : ℝ) (z : Cell)          [a number and a point]
    : Cell                      [the point with both slots stretched] -/
def smul (c : ℝ) (z : Cell) : Cell := (c * z.1, c * z.2)

/-- Stretching both slots multiplies the size by the square of the stretch.

    (σ c : ℝ) (z : Cell)                                   [sigma, a number, a point]
    normForm σ (smul c z) = c ^ 2 * normForm σ z           [the size scales by the square] -/
theorem normForm_smul (σ c : ℝ) (z : Cell) :
    normForm σ (smul c z) = c ^ 2 * normForm σ z := by
  simp only [normForm, smul]
  ring

/-- Stretching both slots of a projective point gives a projective point.

    {σ : ℝ} (p : ProjPair σ)             [a point of the projective line]
    {c : ℝ} (hc : c ≠ 0)                 [a non-zero stretch]
    : ProjPair σ                         [the stretched point, still off the cone] -/
def scale {σ : ℝ} (p : ProjPair σ) {c : ℝ} (hc : c ≠ 0) : ProjPair σ where
  b := smul c p.b
  u := smul c p.u
  u_unit := by
    rw [normForm_smul]
    exact mul_ne_zero (pow_ne_zero 2 hc) p.u_unit

/-- Reading a projective point as one number, channel by channel.

    {σ : ℝ} (p : ProjPair σ)                        [a point of the projective line]
    : ℝ                                             [one number]
    upper σ p.b / upper σ p.u                       [the first channel's reading of the fraction]

    So: the dehomogenisation, taken on the upper channel. `ratio` in
    `Trichotomy/Application.lean` is this construction on a single point rather than on a pair. -/
noncomputable def dehomUpper {σ : ℝ} (p : ProjPair σ) : ℝ := upper σ p.b / upper σ p.u

/-- The reading does not change when both slots are stretched.

    {σ : ℝ} (hσ : 0 ≤ σ) (p : ProjPair σ)               [sigma at zero or above, and a point]
    {c : ℝ} (hc : c ≠ 0)                                [a non-zero stretch]
    dehomUpper (scale p hc) = dehomUpper p              [the reading is unchanged]

    So: the projective fact itself. A projective point is a pair counted up to a common
    factor, and this says the reading respects that counting. The predecessor proved this twice
    over the reals, in `Bulk/ProjectiveStructure.lean` and `Bulk/ScaleInvariance.lean`; they are
    one lemma, and over this algebra it needs the second slot off the cone, not just
    non-zero. -/
theorem dehomUpper_scale {σ : ℝ} (hσ : 0 ≤ σ) (p : ProjPair σ) {c : ℝ} (hc : c ≠ 0) :
    dehomUpper (scale p hc) = dehomUpper p := by
  have hu : upper σ p.u ≠ 0 := (channels_ne_zero hσ p).1
  have hsu : ∀ z : Cell, upper σ (smul c z) = c * upper σ z := by
    intro z
    simp only [upper, smul]
    ring
  simp only [dehomUpper, scale, hsu]
  rw [mul_div_mul_left _ _ hc]

/-- Reading a projective point on the other channel.

    {σ : ℝ} (p : ProjPair σ)                        [a point of the projective line]
    : ℝ                                             [one number]
    lower σ p.b / lower σ p.u                       [the other channel's reading of the fraction] -/
noncomputable def dehomLower {σ : ℝ} (p : ProjPair σ) : ℝ := lower σ p.b / lower σ p.u

/-- At zero the two readings are one reading.

    (p : ProjPair 0)                       [a point of the projective line at the middle face]
    dehomUpper p = dehomLower p            [both channels read the same number]

    So: the parabolic face has one chart where the hyperbolic face has two. This is
    `upper_eq_lower_of_disc_zero` carried up to the projective line, and it is the structural
    reason the Galilean construction in the physics literature needs a vanishing factor divided
    out: at this face the difference of the two readings is identically zero. -/
theorem dehom_agree_of_disc_zero (p : ProjPair 0) : dehomUpper p = dehomLower p := by
  simp only [dehomUpper, dehomLower, upper_eq_lower_of_disc_zero]

-- ═══════════════════════════════════════════════════════════════════
-- THE THREE FACES, AS GEOMETRIES
-- ═══════════════════════════════════════════════════════════════════

/-! Kisil's *Erlangen Program at Large* takes the half-plane over each of the three algebras and
works them as the three geometries of the `SL(2,ℝ)` action: at `σ = −1` the Poincaré half-plane,
which is the hyperbolic plane; at `σ = 0` the Galilean plane; at `σ = +1` the Minkowski
half-plane. One group, three signatures, and sigma decides which. The invariant curves are his
cycles, and he compactified all three in J. Math. Phys. 48 073506, where the hyperbolic case
needs a time-arrow condition to come out right.

Three theorems already in this corpus say what each face supplies to the projective reading:

- `no_channel_of_disc_neg`: below zero there is no channel reading at all, so the two-channel
  chart does not exist and the algebra is the definite one
- `upper_eq_lower_of_disc_zero`, carried up as `dehom_agree_of_disc_zero`: at zero the two
  readings coincide, so there is one chart where above zero there are two
- `two_channels_iff_disc_pos`: above zero there are two channel readings and only two

What this corpus adds to Kisil is that the family is **complete**. `classification` proves every
two-dimensional unital commutative associative real algebra is one of the three, machine-checked
on `[propext, Classical.choice, Quot.sound]`. His programme states the three and develops them;
it does not prove there is no fourth. That is the half a proof assistant is for.
-/

-- ═══════════════════════════════════════════════════════════════════
-- THE MOBIUS ACTION
-- ═══════════════════════════════════════════════════════════════════

/-- A two-by-two real matrix acting on a pair of points.

    (M : Matrix (Fin 2) (Fin 2) ℝ)          [a two-by-two grid of numbers]
    (q : Cell × Cell)                       [a pair of points]
    : Cell × Cell                           [the pair recombined by the grid]

    So: the linear action underneath a Möbius transformation. Reading the result as one number
    turns it into `(a·z + b)/(c·z + d)`, which is the form Kisil's `SL(2,ℝ)` acts by, but the
    linear form is total where the fraction is not. -/
def act (M : Matrix (Fin 2) (Fin 2) ℝ) (q : Cell × Cell) : Cell × Cell :=
  (M 0 0 • q.1 + M 0 1 • q.2, M 1 0 • q.1 + M 1 1 • q.2)

/-- The identity grid changes nothing.

    (q : Cell × Cell)          [a pair of points]
    act 1 q = q                [acting by the identity leaves it alone] -/
theorem act_one (q : Cell × Cell) : act 1 q = q := by
  simp [act]

/-- Acting by a product is acting twice.

    (M N : Matrix (Fin 2) (Fin 2) ℝ)          [two grids]
    (q : Cell × Cell)                         [a pair of points]
    act (M * N) q = act M (act N q)           [their product acts as one then the other]

    So: this is the group action law, and it is what makes the two-by-two grids a symmetry of the
    projective line rather than a collection of unrelated maps. The predecessor named this in the
    title of `Bulk/ProjectiveStructure.lean`, imported Mathlib's matrix determinant and inverse
    files for it, and never stated it; its own header records that no `Matrix` appears in any of
    its declarations. -/
theorem act_mul (M N : Matrix (Fin 2) (Fin 2) ℝ) (q : Cell × Cell) :
    act (M * N) q = act M (act N q) := by
  simp only [act, Matrix.mul_apply, Fin.sum_univ_two, Prod.ext_iff]
  constructor <;> · simp only [add_smul, smul_add, smul_smul]; abel_nf; trivial

/-- Where the chart breaks: acting can land the second slot on the cone.

    {σ : ℝ} (M : Matrix (Fin 2) (Fin 2) ℝ) (p : ProjPair σ)   [sigma, a grid, a point]
    normForm σ (act M (p.b, p.u)).2 ≠ 0                       [the image's second slot is off it]
    → ProjPair σ                                              [then the image is a point again]

    So: the action is total on pairs and partial on the chart. A Möbius transformation may send a
    point to where the fraction is undefined, which over the reals is the single point at
    infinity and here is the whole null cone, two lines by `normForm_zero_iff_channel_zero`.
    Naming the condition is what the structure is for. -/
def actChart {σ : ℝ} (M : Matrix (Fin 2) (Fin 2) ℝ) (p : ProjPair σ)
    (h : normForm σ (act M (p.b, p.u)).2 ≠ 0) : ProjPair σ where
  b := (act M (p.b, p.u)).1
  u := (act M (p.b, p.u)).2
  u_unit := h

end Projective
end Apeiron
