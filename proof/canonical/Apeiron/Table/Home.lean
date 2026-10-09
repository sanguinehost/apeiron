import Apeiron.Trichotomy.NullBasis

/-!
# The hashtrinity readout: one channel decides, the other is a genuine key

@name Table Home
@grade proved

Ported from `../sanguine-ref/Substrate/Algebra/NullCell/SpinorFold.lean`, at σ general rather
than the predecessor's fixed σ = 1: its `NullCell` carrier is `nNorm z = z.1² − z.2²`, which is
`Trichotomy.normForm 1`, so its whole file sits at that one value of σ. `Trichotomy.upper`,
`Trichotomy.lower` and `Trichotomy.ofHalves` already generalise the change of basis to every
σ > 0, and `Trichotomy.halves_ext` already proves that both coordinates together decide a
point; the two theorems below are what a decryptor on ONE coordinate buys over that, and the
two after them are a small fact about hash-table homes the same reading names.

`one_half_decides`: when the second coordinate is a function of the first, agreeing on the
first coordinate alone already forces the two points equal. This is `hashtrinity`'s `get`: the
coordinate a lookup reads is `upper`, the key a naive table would also store is `lower`, and a
coordinate key's `lower` is recoverable from `upper` (`dec`), so storing it is redundant.
`opaque_needs_lower` is the dual: without that recoverability, two points sharing `upper` are
not forced equal, so their `lower` must differ if the points do, and the second
coordinate cannot be dropped.

`coord_home_collisionFree` and `opaque_home_collides` are the write-side fact the same reading
names: a coordinate key's own value as its home (`home = id`) never collides with another,
while a truncating fold (`home = (· >>> 1)`) can, at every load. Neither theorem depends on the
algebra above; they are stated because the source file's own reading pairs them with it.

## What is not ported

The source file's own Next and Anchor bullets say what its results do not establish, and
neither is carried here. It names an operation count, "`nmul` is four real products,
`nmulSpinor` is two, so schedule-free reduction is `2×` faster", as `Next`: read off the
shapes of two definitions, not counted by any theorem, and not ported. It names "this 2× is the
covering degree of `Spin(2) → SO(2)`" as `Anchor`: no `Spin(2)`, no `SO(2)`, no covering degree
occurs in the source file or in this one, and neither does the topological reading built on
them. The source file's `spinorFold`, `nmul_is_diagonal`, `rotor_folds_into_weight` and the
"trinity" section (`monad_is_the_third`, `nulls_are_the_two_single_halves`) are not part of
this port either: the task this file discharges is the hashtrinity readout alone, and the rest
of that file's diagonalisation content already has a home in
`Trichotomy/NullBasis.lean`'s `upper_mul`/`lower_mul`/`normForm_eq_upper_mul_lower`, proved
there for every σ, not restated here.
-/

namespace Apeiron
namespace Table

open Trichotomy

/-- **One coordinate decides, given a decryptor.** When the second coordinate is some function
    of the first on both points, agreeing on the first coordinate alone forces the points
    equal; the second coordinate carries nothing `halves_ext` needed and did not already get
    from the first.

    {σ : ℝ} (hσ : 0 < σ)                              [the parameter, above zero]
    {z w : Cell}                                      [two points]
    (dec : ℝ → ℝ)                                     [a function recovering one coordinate
                                                       from the other]
    (hz : lower σ z = dec (upper σ z))                [the first point's second coordinate is
                                                       `dec` of its first]
    (hw : lower σ w = dec (upper σ w))                [and so is the second point's]
    (h : upper σ z = upper σ w)                       [the two points agree on the first
                                                       coordinate]
    z = w                                             [they are the same point]

    `Cell` [a point: two real numbers written side by side, a + b·x]
    `upper` [how far a point lies along the first direction of size zero]
    `lower` [how far a point lies along the other direction of size zero]
    `ℝ` [the real numbers: every number on the number line]
    `→` [if the left holds so does the right]

    So: this is `hashtrinity`'s `get`. `upper` is the coordinate a lookup reads; `lower` is the
    key a naive table would also store; a coordinate key's `lower` is `dec (upper ·)` for a
    fixed `dec`, so the stored key is redundant and the read alone decides identity. -/
theorem one_half_decides {σ : ℝ} (hσ : 0 < σ) {z w : Cell} (dec : ℝ → ℝ)
    (hz : lower σ z = dec (upper σ z)) (hw : lower σ w = dec (upper σ w))
    (h : upper σ z = upper σ w) : z = w :=
  halves_ext hσ h (by rw [hz, hw, h])

/-- **The dual: without a decryptor, the second coordinate must be carried.** Two distinct
    points sharing their first coordinate cannot also share their second, or `halves_ext` would
    force them equal. So absent a `dec` recovering one coordinate from the other, agreement on
    `upper` alone proves nothing, and `lower` is not redundant.

    {σ : ℝ} (hσ : 0 < σ)                              [the parameter, above zero]
    {z w : Cell}                                      [two points]
    (hu : upper σ z = upper σ w)                      [they agree on the first coordinate]
    (hne : z ≠ w)                                     [and are nonetheless distinct points]
    lower σ z ≠ lower σ w                             [their second coordinates differ]

    `Cell` [a point: two real numbers written side by side, a + b·x]
    `upper` [how far a point lies along the first direction of size zero]
    `lower` [how far a point lies along the other direction of size zero]
    `≠` [is not equal to]

    So: an opaque key with no decryptor must carry both coordinates; only a coordinate key with
    a genuine `dec` licenses dropping the second (`one_half_decides`). -/
theorem opaque_needs_lower {σ : ℝ} (hσ : 0 < σ) {z w : Cell}
    (hu : upper σ z = upper σ w) (hne : z ≠ w) : lower σ z ≠ lower σ w :=
  fun hl => hne (halves_ext hσ hu hl)

/- ── the write side: does the home function collide? ── -/

/-- Two keys collide at a home function: distinct keys, the same home slot.

    (home : ℕ → ℕ)                    [a function from keys to home slots]
    (a b : ℕ)                         [two keys]
    : Prop                            [whether they collide there]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on] -/
def collides (home : ℕ → ℕ) (a b : ℕ) : Prop := a ≠ b ∧ home a = home b

/-- **A coordinate key's own value as its home never collides.** `home = id` is injective, so
    distinct keys never share a slot, at any table size: the probe that would resolve a
    collision has nothing to resolve.

    (a b : ℕ)                         [two keys]
    ¬ collides id a b                 [they do not collide there]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `collides` [a function from keys to home slots, and two keys, to whether they collide there]
    `¬` [this is not true] -/
theorem coord_home_collisionFree (a b : ℕ) : ¬ collides id a b :=
  fun ⟨hne, heq⟩ => hne heq

/-- **A truncating fold can collide.** `(· >>> 1)`, one bit dropped, sends two distinct keys to
    the same slot: `0` and `1` both land on `0`. A non-injective home needs the probe the
    coordinate case never runs.

    ∃ a b, collides (· >>> 1) a b     [some pair of keys collides under the truncating fold]

    `∃` [there is at least one]
    `collides` [a function from keys to home slots, and two keys, to whether they collide there] -/
theorem opaque_home_collides : ∃ a b, collides (· >>> 1) a b :=
  ⟨0, 1, by decide, by decide⟩

end Table
end Apeiron
