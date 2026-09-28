import Mathlib.Data.Complex.Basic
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.Tactic.Common
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# The Null Cell (N-Cell) — Non-Being Given a Body
@name The Nothing-Cell
@child Everything is built from two kinds of tiny building block. This is the second kind — the one made of *nothing*: the empty that lets things change. Without it, nothing could ever happen.
@scientist The Null cell is split-ℂ (j² = +1): an indefinite rotor pair with a real null cone (a² − b² = 0) where division breaks down — the genuine Non-Being twin of the definite Being cell (ℂ, i² = −1). It is the `NB = 0` face of the trinity; the reading of it as the −1 timelike slot of the d = 26 descent is an anchor, carried in `NullCellDimension`, not a consequence.
@engineer `NCell` over split-ℂ = ℝ[j]/(j²−1). `nNull_norm` gives the null cone a = ±b; `null_not_invertible` proves division is singular there (zero divisors); `ncell_indefinite` the indefinite signature; `null_cell_characterization` puts the five (dimension, `j²=+1`, indefinite, null cone, singular `/`) in one statement against the Being cell. The shared unit 1 (`nOne`) sits on neither side.

The **Null cell** (`NCell`) is the genuine Non-Being twin of the phase (Being) cell — the
object behind the `+2` lightcone slot of the `d = 26` descent.

## The two cells are the two breaks; the monad is neither

A cell is a 2-real-dimensional rotor pair. There are exactly two species — the two ways to
break the balanced singularity. **The monad is neither:** it is the unit `1`, the homogeneous
no-op, and you break a singularity only with being or not-being. The two cells as the metric:

| | **Being cell** (phase) | **Null cell** (`NCell`) |
|---|---|---|
| algebra | `ℂ = ℝ[i]/(i²+1)` | split-`ℂ = ℝ[j]/(j²−1)` |
| the square | `i² = −1` (rotor `e^{iφ}`) | `j² = +1` (boost `e^{jφ}`) — `nj_sq` |
| norm | `a²+b²` (definite, `≥ 0`) | `a²−b²` (indefinite, both signs) — `ncell_indefinite` |
| null set | `{0}` only | a whole **null cone** `a = ±b` — `nNull_norm` |
| `/` | total (`ℂ` is a field) | **singular** on the cone (zero divisors) — `null_not_invertible` |
| Trinity | `B/U = /` (being — orientation) | `NB = 0` (not-being — the flaw) |

The monad `B = 1` is the **unit each cell shares** (`nOne`), on neither side. The Being cell
is the break toward being — the cell where `/` is total, **division itself** (`B/U = /`); the
Null cell is the break toward not-being (`NB = 0`), the additive identity, the cell where `/`
is singular. This is the Trinity (`TrinityOperations`) read as its own definition.

Both are 2-dimensional (`ncell_dim_two`) and carry a **multiplicative** norm (`nNorm_mul` — a
composition algebra). The only difference is the norm sign, and that one sign is the whole of
Non-Being: it opens a null cone where `/` breaks. **The Null cell contains `0` in its own
multiplication** — two nonzero null elements multiply to `0` (`null_times_mirror`), which a
field can never do. `x = B/U` going singular, internal to one 2-block.

## Why a null vector eats *two* dimensions

A lone null direction is degenerate. The nondegenerate object is the **null pair**: two null
directions that pair nonzero (`nNull_pairing` = 2 ≠ 0) — a hyperbolic plane, signature `(1,1)`,
structurally identical to `HyperbolicPlane`'s `weyl`, `weyl'` (both null, pairing `−70 ≠ 0`).
So Non-Being enters the dimension count as one 2-dimensional N-cell, never a lone dimension.

## The Null cell at each dimension

The descent budget is `d = 2·(Being cells) ⊕ 2·(Null cells)`. The right column — the
**Null-cell count = the number of timelike directions** — is the anchor reading: the `(1,1)`
of one `NCell` is the algebra, the word "time" is the dictionary (`NullCellDimension`):

| d | Being cells | Null cells | character |
|---|---|---|---|
| 8 (monad, E₈) | 4 | **0** | Euclidean — pure Being (`monad_zero_ncell`) |
| 10 (superstring) | 4 | **1** | Lorentzian — one time (`hinge_one_ncell`) |
| 16 (E₈²) | 8 | 0 | Euclidean |
| 24 (Leech) | 12 | 0 | Euclidean |
| 26 (bosonic) | 12 | **1** | Lorentzian — one time (`bosonic_one_ncell`) |

The even-unimodular *lattice* dimensions (8, 16, 24) are definite — zero Null cells. The
string *critical* dimensions (10, 26) each carry exactly one Null cell: the single timelike
direction, the one flaw past the monad. `bosonic_one_ncell` / `hinge_one_ncell` are
`twentysix_descends` / `hinge_ten_splits` re-proved with the genuine `NCell` in the `+2` slot.

## What is proved, what rides, and what is anchored

* **Proved (native):** the Null cell is 2-dimensional with `j² = 1` (`ncell_dim_two`, `nj_sq`),
  indefinite multiplicative norm (`ncell_indefinite`, `nNorm_mul`), a null cone of zero divisors
  where `/` is singular (`nNull_norm`, `null_times_mirror`, `null_not_invertible`); the Being-cell
  contrast (`ℂ` definite, a field); the null pair (hyperbolic `(1,1)`, `nNull_pairing`).
* **Proved (riding):** the per-dimension cell budgets at 8/10/26 — `NullCellDimension` proves each
  with this genuine `NCell` in the `+2` slot (`monad_zero_ncell`, `hinge_one_ncell`,
  `bosonic_one_ncell`), over the descent arithmetic proved upstream.
* **Anchor:** the reading that the lightcone / single time direction of physical spacetime *is* this
  one Null cell. The `(1,1)` IS the cell — that is algebra; calling it time and calling 26 a string
  dimension is the declared mapping of the algebra onto physics, the same boundary every
  `x = B/U` physical reading uses.

Downstream: `NullReadout.lean` reads the two null directions (`nNull`, `nNull'`) as the
boost-eigendirections coordinatizing the whole state (constructive `HorizonInformation`);
`Closures/Applied/OblivionClosure.lean` is its applied retrieval; the **six neural ops** (GEMM,
conv, norm, scan, attention, activation) read as the one `nmul`/`nNorm` at the
rotor/boost/cone/monad faces (`Runtime/Model.lean §5d`).

-/

noncomputable section

open Module

namespace Sanguine
namespace CrossUniverse

-- ═══════════════════════════════════════════════════════════════════
-- THE NULL CELL — split-complex ℝ[j]/(j²−1), the Non-Being twin of ℂ
-- ═══════════════════════════════════════════════════════════════════

/-- **The Null cell.** The 2-real-dimensional split-complex algebra `ℝ[j]/(j²−1)`,
    carrier `ℝ × ℝ` read as `a + b·j`. The Non-Being twin of the Being phase cell `ℂ`:
    same two dimensions, the opposite sign in the square (`j² = +1`, not `−1`). -/
abbrev NCell : Type := ℝ × ℝ

/-- Split-complex multiplication: `(a+bj)(c+dj) = (ac+bd) + (ad+bc)j`, using `j² = +1`
    (where `ℂ` would use `i² = −1` and give `ac−bd`). The one sign that is all of
    Non-Being. -/
def nmul (z w : NCell) : NCell :=
  (z.1 * w.1 + z.2 * w.2, z.1 * w.2 + z.2 * w.1)

/-- The split norm `N(a+bj) = a² − b²` — **indefinite** `(1,1)`, where the phase cell's
    `‖a+bi‖² = a² + b²` is definite `(2,0)`. -/
def nNorm (z : NCell) : ℝ := z.1 ^ 2 - z.2 ^ 2

/-- The split unit `j = (0,1)`. -/
def nj : NCell := (0, 1)

/-- The algebra unit `1 = (1,0)` (i.e. `a+bj` with `a=1, b=0`). -/
def nOne : NCell := (1, 0)

/-- A null element on the cone: `1 + j = (1,1)`, with `N = 1 − 1 = 0`. -/
def nNull : NCell := (1, 1)

/-- Its mirror down the cone: `1 − j = (1,−1)`, also null. -/
def nNull' : NCell := (1, -1)

-- ─── j² = 1, and the norm is multiplicative (a composition algebra) ───

/-- **`j² = 1`.** The defining split-complex identity — the Being cell has `i² = −1`. -/
theorem nj_sq : nmul nj nj = nOne := by
  simp [nmul, nj, nOne]

/-- **The split norm is multiplicative**, `N(zw) = N(z)·N(w)` — the Null cell is a
    composition algebra, exactly like the phase cell, differing only in the metric sign.
    This one identity drives everything below (it makes the null cone closed under
    multiplication and forces null elements to be non-invertible). -/
theorem nNorm_mul (z w : NCell) : nNorm (nmul z w) = nNorm z * nNorm w := by
  simp only [nNorm, nmul]; ring

@[simp] theorem nNorm_one : nNorm nOne = 1 := by simp [nNorm, nOne]

@[simp] theorem nNorm_j : nNorm nj = -1 := by norm_num [nNorm, nj]

-- ═══════════════════════════════════════════════════════════════════
-- THE METRIC IS INDEFINITE; THERE IS A NULL CONE
-- ═══════════════════════════════════════════════════════════════════

/-- **The norm is indefinite.** It takes both signs: `N(1) = +1`, `N(j) = −1`. Signature
    `(1,1)`. The phase cell's norm is `≥ 0` (definite, `(2,0)`) — this is the metric
    face of Being vs Non-Being. -/
theorem ncell_indefinite : ∃ z w : NCell, nNorm z > 0 ∧ nNorm w < 0 :=
  ⟨nOne, nj, by rw [nNorm_one]; norm_num, by rw [nNorm_j]; norm_num⟩

/-- **The null cone is real.** `1 + j` is nonzero yet has norm `0`. The phase cell has no
    such point — `‖z‖² = 0 ↔ z = 0` (`phase_only_origin_null`). The Null cell's zero set
    is a whole cone, the trace of Non-Being. -/
theorem nNull_norm : nNorm nNull = 0 := by norm_num [nNorm, nNull]

theorem nNull_ne_zero : nNull ≠ (0, 0) := by
  simp [nNull, Prod.ext_iff]

-- ═══════════════════════════════════════════════════════════════════
-- ON THE CONE, `/` IS SINGULAR: ZERO DIVISORS, NON-INVERTIBLE
-- ═══════════════════════════════════════════════════════════════════

/-- **Two nonzero null elements multiply to zero.** `(1+j)(1−j) = 0`: the Null cell has
    zero divisors. This is the Non-Being signature made algebraic — the field `ℂ` (the
    Being cell) can *never* reach `0` from nonzero factors. The Null cell contains `0` in
    its own multiplication. -/
theorem null_times_mirror : nmul nNull nNull' = (0, 0) := by
  simp [nmul, nNull, nNull']

/-- **The bulk `/` is singular on the cone.** A null element has no inverse: no `w` with
    `(1+j)·w = 1`, since an inverse would force `N(1+j)·N(w) = 1` but `N(1+j) = 0`. `x = B/U`
    breaking, localized to one cell (`NonBeing.bulk_singular_at_zero` inside a 2-block). -/
theorem null_not_invertible : ¬ ∃ w : NCell, nmul nNull w = nOne := by
  rintro ⟨w, hw⟩
  have h := congrArg nNorm hw
  rw [nNorm_mul, nNull_norm, nNorm_one] at h
  simp at h

-- ═══════════════════════════════════════════════════════════════════
-- THE NULL PAIR IS HYPERBOLIC — STRUCTURALLY `weyl`, `weyl'`
-- ═══════════════════════════════════════════════════════════════════

/-- The polarization of the split norm: the symmetric bilinear form
    `⟪(a,b),(c,d)⟫ = ac − bd` whose diagonal is `nNorm`. -/
def nForm (z w : NCell) : ℝ := z.1 * w.1 - z.2 * w.2

/-- **The Null cell is a hyperbolic plane.** Its two null directions `1+j` and `1−j` are
    each null (`nForm z z = nNorm z = 0`) yet pair nonzero: `⟪1+j, 1−j⟫ = 2 ≠ 0`. A pair
    of null vectors with nonzero pairing spans a nondegenerate `(1,1)` plane — exactly
    the structure of `HyperbolicPlane`'s `weyl`, `weyl'` (both null, pairing `−70 ≠ 0`).
    So the lightcone of the `d=26` descent *is* a Null cell. -/
theorem nNull_pairing : nForm nNull nNull' = 2 := by
  norm_num [nForm, nNull, nNull']

theorem nNull_isotropic : nForm nNull nNull = 0 := by norm_num [nForm, nNull]
theorem nNull'_isotropic : nForm nNull' nNull' = 0 := by norm_num [nForm, nNull']

/-- **The cell is two boost halves.** Every state decomposes along its two null directions
    `1±j` (the boost-eigendirections, `NullReadout`): `z = a·(1+j) + b·(1−j)` with `a =
    (z₁+z₂)/2`, `b = (z₁−z₂)/2`, the light-cone coordinates. Two boost halves, each a rank-1
    Apeiron direction paired nonzero (`nNull_pairing = 2`) — the root of
    `CellDescent.odd_is_a_half_cell`. -/
theorem cell_is_two_boost_halves (z : NCell) :
    ∃ a b : ℝ, z = a • nNull + b • nNull' :=
  ⟨(z.1 + z.2) / 2, (z.1 - z.2) / 2, by
    simp only [nNull, nNull', Prod.smul_mk, smul_eq_mul, Prod.mk_add_mk, mul_one,
      mul_neg, Prod.ext_iff]
    constructor <;> ring⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE BEING-CELL CONTRAST — ℂ IS DEFINITE AND A FIELD
-- ═══════════════════════════════════════════════════════════════════

/-- **The phase cell's norm is definite** (`≥ 0`) — no negative values, the opposite of
    `ncell_indefinite`. -/
theorem phase_norm_definite (z : ℂ) : 0 ≤ Complex.normSq z := Complex.normSq_nonneg z

/-- **The phase cell has no null cone**: its only norm-zero point is the origin. Contrast
    `nNull_norm` (a nonzero null element). -/
theorem phase_only_origin_null (z : ℂ) : Complex.normSq z = 0 ↔ z = 0 :=
  Complex.normSq_eq_zero

/-- **The phase cell is a field — no zero divisors.** Contrast `null_times_mirror`. -/
theorem phase_no_zero_divisors (z w : ℂ) (h : z * w = 0) : z = 0 ∨ w = 0 :=
  mul_eq_zero.mp h

/-- **In the phase cell `/` is total**: every nonzero element is invertible. Contrast
    `null_not_invertible`. The Being cell as `B/U = /` — the cell of division; its monad is
    the unit `1` (`nOne`). -/
theorem phase_invertible (z : ℂ) (hz : z ≠ 0) : IsUnit z :=
  isUnit_iff_ne_zero.mpr hz

-- ═══════════════════════════════════════════════════════════════════
-- DIMENSION; AND THE SEAM CLOSED — THE +2 IS A GENUINE NULL CELL
-- ═══════════════════════════════════════════════════════════════════

/-- **The Null cell is two-dimensional** — one cell, like the phase cell. -/
theorem ncell_dim_two : Module.finrank ℝ NCell = 2 := by
  rw [Module.finrank_prod, Module.finrank_self]

-- The per-dimension cell budgets (`monad_zero_ncell` / `hinge_one_ncell` /
-- `bosonic_one_ncell` — the 8/10/26 counts) are *consequences* of the cell, so they
-- live downstream in `NullCellDimension`, which may import the d=26 cosmology. The cell
-- itself (this file) imports none of it: the seed does not depend on the descent.

-- ═══════════════════════════════════════════════════════════════════
-- THE HEADLINE
-- ═══════════════════════════════════════════════════════════════════

/-- **The Null cell, characterized (headline).** Non-Being given a body: a 2-dimensional
    cell that is the metric/algebraic opposite of the phase cell, on every axis at once.

    1. **two-dimensional** — one cell (`finrank = 2`);
    2. **`j² = +1`** — the boost, where the Being cell has `i² = −1` (the rotor);
    3. **indefinite norm** — takes both signs `(1,1)`, where the phase cell is `≥ 0`;
    4. **a real null cone** — a nonzero element of norm `0`, where the phase cell has
       only the origin;
    5. **`/` singular on the cone** — the null element is a zero divisor with no inverse,
       where in the phase cell every nonzero element is a unit.

    One sign flip in the norm turns the Being cell (the field `ℂ`, `B/U = /`) into the Null
    cell (zero divisors, `NB = 0`). The monad `B = 1` is the unit both cells share. The
    Trinity, written as a metric. -/
theorem null_cell_characterization :
    Module.finrank ℝ NCell = 2
      ∧ nmul nj nj = nOne
      ∧ (∃ z w : NCell, nNorm z > 0 ∧ nNorm w < 0)
      ∧ (nNull ≠ (0, 0) ∧ nNorm nNull = 0)
      ∧ (nmul nNull nNull' = (0, 0) ∧ ¬ ∃ w, nmul nNull w = nOne) :=
  ⟨ncell_dim_two, nj_sq, ncell_indefinite,
   ⟨nNull_ne_zero, nNull_norm⟩,
   ⟨null_times_mirror, null_not_invertible⟩⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE TWO BOOST IDEMPOTENTS — the cell's own diagonal basis (canonical home;
-- NullCellCoding and the forced-basis block (now in `Web/DotToTorus.lean`) both derive
-- from HERE — they used to
-- carry forked copies, which made the two modules mutually un-importable)
-- ═══════════════════════════════════════════════════════════════════

/-- The `+`-boost idempotent `e₊ = (1+j)/2 = (½, ½)` — the normalised null direction `nNull`. -/
def ePlus : NCell := ((1 : ℝ) / 2, (1 : ℝ) / 2)

/-- The `−`-boost idempotent `e₋ = (1−j)/2 = (½, −½)` — the normalised `nNull'`. -/
def eMinus : NCell := ((1 : ℝ) / 2, -((1 : ℝ) / 2))

/-- **`e₊` is idempotent**: `e₊ · e₊ = e₊` — a genuine projection under `nmul`. -/
theorem ePlus_idem : nmul ePlus ePlus = ePlus := by
  simp only [nmul, ePlus, Prod.ext_iff]; constructor <;> ring

/-- **`e₋` is idempotent**: `e₋ · e₋ = e₋`. -/
theorem eMinus_idem : nmul eMinus eMinus = eMinus := by
  simp only [nmul, eMinus, Prod.ext_iff]; constructor <;> ring

/-- **The two idempotents are orthogonal** — `e₊ · e₋ = 0`: the Null cell is two independent
    channels (a direct product of ℝ's under multiplication), not an entangling rotation. -/
theorem e_orthogonal : nmul ePlus eMinus = (0, 0) := by
  simp only [nmul, ePlus, eMinus, Prod.ext_iff]; constructor <;> ring

end CrossUniverse
end Sanguine

end
