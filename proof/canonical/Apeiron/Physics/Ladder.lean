import Apeiron.Trichotomy.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

/-!
# Counting cells, and what the remainder counts

@name Ladder
@cite conway-sloane-1999
@cite watson-1918
@grade proved

Build a space out of two kinds of block. One block is eight real directions of definite size, the
face where nothing but the origin has size zero. The other is one cell of indefinite size, two real
directions carrying a pair of null directions that pull on each other.
`Trichotomy.hasNullPair_iff_disc_pos` says the second kind exists above zero and nowhere else, so
the two block kinds are the two sides of `σ = 0` and not two conventions.

A space of `e` definite blocks and `u` indefinite cells has `8e + 2u` real directions, and the
remainder on division by eight counts the cells:

```
   dimOf e u % 8  =  2 · (u % 4)          (residue_counts_null_pairs)
```

so with at most three cells the count is read straight off the dimension
(`null_count_from_residue`). One cell puts the dimension at two more than a multiple of eight,
which is `10, 18, 26` (`one_cell_column`).

## What this file settles and what it does not

The counting here is arithmetic plus two dimension facts, and it is not a classification.

Both coefficients are given content downstream. `Apeiron.Lattice.EvenUnimodular` constructs
`E₈^e ⊕ U^u`, even and unimodular of rank `dimOf e u`, so every entry of the table is realised by
a lattice; `Apeiron.Lattice.hyperbolicPlane_gram_is_the_null_pair` shows the rank-two summand's
Gram matrix is the matrix of `Trichotomy.form 1` on the null pair, so `u` counts `σ > 0` null
pairs by theorem; and `Apeiron.Doubling.ladder_period_is_octonion` writes the `8` as
`Module.finrank ℝ Doubling.Octonion`, obtained by doubling rather than asserted.

What stays open is the converse: that every even unimodular form of signature `(p, q)` has
`p − q` divisible by eight is van der Blij's congruence, which is cited and not proved.
`theory/DEAD_ENDS.md` records the three routes to it and the Mathlib primitive each one needs.

## Root

Rides `Trichotomy.hasNullPair_iff_disc_pos` for which face supplies an indefinite cell, and
`Trichotomy.exists_normForm_zero_iff_disc_nonneg` for which faces have any null direction at all.
The edge:

* `dimOf_as_cells` writes the ladder's two coefficients as `Module.finrank ℝ Cell` and four times
  it, so the `2` and the `8` are the algebra's own dimension rather than chosen constants. The
  remainder law itself is arithmetic and `σ` does not appear in it; where `σ` appears is
  in which face each block is, which is `null_block_iff_disc_pos`.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- **A cell carries a null pair exactly above zero.**

    σ                                   [the parameter]
    ℝ                                   [the real numbers: every number on the number line]
    ↔                                   [exactly when, true in both directions]
    <                                   [is smaller than]
    HasNullPair                         [holds two directions of size zero that still pull]

    So: the indefinite block of the ladder is the face above zero, and the definite block is the
    face below it. The two block kinds are the two sides of `σ = 0`. -/
theorem null_block_iff_disc_pos (σ : ℝ) : HasNullPair σ ↔ 0 < σ :=
  hasNullPair_iff_disc_pos σ

/-- A cell has two real directions.

    Module.finrank                      [how many independent directions the space has]
    ℝ                                   [the real numbers: every number on the number line]
    Cell                                [a point: two real numbers written side by side] -/
theorem cell_finrank : Module.finrank ℝ Cell = 2 := by
  simp [Cell]

/-- The dimension of a space built from the two blocks.

    ℕ                                   [the counting numbers 0, 1, 2, 3, and so on]
    8 * e + 2 * u                       [eight per definite block, two per indefinite cell]
    dimOf                               [the ladder's dimension count] -/
def dimOf (e u : ℕ) : ℕ := 8 * e + 2 * u

/-- **The two coefficients are the cell's own dimension.**

    ℕ                                                          [the counting numbers]
    dimOf                                                      [the ladder's dimension count]
    Module.finrank                                             [how many independent directions]
    ℝ                                                          [the real numbers]
    Cell                                                       [a point: two reals side by side]

    So: the `2` is one cell and the `8` is four of them, so neither coefficient is fitted. -/
theorem dimOf_as_cells (e u : ℕ) :
    dimOf e u = 4 * e * Module.finrank ℝ Cell + u * Module.finrank ℝ Cell := by
  rw [cell_finrank]
  unfold dimOf
  ring

/-- **The remainder on division by eight is twice the cell count.**

    ℕ                                                          [the counting numbers]
    dimOf                                                      [the ladder's dimension count]

    So: the definite blocks vanish on division by eight and the cells do not, so the remainder is
    a count of cells. -/
theorem residue_counts_null_pairs (e u : ℕ) : dimOf e u % 8 = 2 * (u % 4) := by
  unfold dimOf; omega

/-- **With at most three cells the count is read off the dimension.**

    ℕ                                                          [the counting numbers]
    <                                                          [is smaller than]
    dimOf                                                      [the ladder's dimension count] -/
theorem null_count_from_residue (e u : ℕ) (hu : u < 4) : dimOf e u % 8 / 2 = u := by
  unfold dimOf; omega

/-- **One cell puts the dimension two above a multiple of eight.**

    ℕ                                                          [the counting numbers]
    ∧                                                          [both are true]
    dimOf                                                      [the ladder's dimension count]

    So: `10`, `18`, `26`. Which of them a physical theory sits in is not settled here, and neither
    is whether these are all the dimensions with one cell. -/
theorem one_cell_column : 10 = dimOf 1 1 ∧ 18 = dimOf 2 1 ∧ 26 = dimOf 3 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> decide

/-- **No definite blocks means no null direction at all.**

    ℕ                                                          [the counting numbers]
    ∧                                                          [both are true]
    dimOf                                                      [the ladder's dimension count] -/
theorem no_cell_column : 8 = dimOf 1 0 ∧ 16 = dimOf 2 0 ∧ 24 = dimOf 3 0 := by
  refine ⟨?_, ?_, ?_⟩ <;> decide

end Physics
end Apeiron
