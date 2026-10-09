--//@ file: Apeiron/Physics/Flavour.lean
--//@ does: Koide mass quotient and three-generation flavour structure from elliptic face
--//@ ring: R0
--//@ onto: observation:physics:koide_flavour
--//@ cite: Koide (1982), Brannen (2006)
--//@ grade: proved

import Apeiron.Trichotomy.Roots
import Apeiron.Trichotomy.Flow
import Apeiron.Physics.Wick
import Mathlib.Analysis.Complex.Trigonometric

/-!
# Three phases, one amplitude, and what the mass cone leaves free

@name Flavour
@cite koide-1982
@cite foot-1994
@cite koide-2000
@cite brannen-2006
@cite sumino-2009
@cite bhatt-kaushik-raj-singh-2022
@grade proved

Three generations is three symmetric copies of one thing, and
`Trichotomy.exists_cube_root_one_iff_disc_neg` says the algebra holds an element of order three
exactly below zero. So a three-fold flavour structure carried by the algebra is on the elliptic
face and on neither of the others, and the element is pinned: its first slot is `−1/2` with no
freedom left, which is the third of a turn (`third_turn_is_the_cube_root`). The three phases are
not chosen; they are the only ones available.

Sample one amplitude at those three phases and add a common size:

```
   x_k  =  M · (1 + a · cos(δ + 2πk/3))          k = 0, 1, 2
```

Then two identities do all the work. The three cosines sum to zero (`trinity_cos_sum`), so the
size survives untouched in `x₀ + x₁ + x₂ = 3M`. Their squares sum to `3/2`
(`trinity_cos_sq_sum`), so the amplitude enters the sum of squares once. Feed both into

```
   koideQuotient x y z  =  (x² + y² + z²) / (x + y + z)²
```

and the size and the phase both cancel: `koide_of_trinity` gives `(2 + a²)/6`, so
`koide_iff_amplitude_sq_two` says the quotient is `2/3` exactly when `a² = 2`.

The same quotient has a geometric reading. `cone_law` says `coneAngleSq · koideQuotient = 1/3`,
so the quotient is one over three times the squared cosine of the angle the vector of square
roots makes with the all-equal direction, and `2/3` is exactly the half-turn's half, `45°`
(`koide_is_quarter_turn`).

What is left free is one number. `azimuth_invariant` says the product of the three cosines is
`cos 3δ / 4`, and `trinity_root` says all three are roots of the one cubic `4c³ − 3c = cos 3δ`,
so with the amplitude pinned the whole spectrum is the single quantity `cos 3δ`, modulo the size.
Nothing in this corpus fixes it.

## What is claimed elsewhere, and what is not

The sweep of `theory/LITERATURE.md` field F records the verdicts. The cone reading is Foot,
arXiv:hep-ph/9402242 (1994), equations (2) and (3). The three-phase form and the quotient
`(2 + a²)/6` are Koide, arXiv:hep-ph/0005137 (2000), equations (1.5) and (2.5), and Brannen
(2006), equations (3), (4) and (7), who derives them as the eigenvalues of a circulant matrix.
Reading the amplitude off each sector's own quotient is Koide's Table 1. All of that is ported
here, cited, and named for what it proves rather than for the reading it carries.

Unclaimed in that sweep: `azimuth_invariant` and `trinity_root`, the reduction of the remaining
freedom to one cubic invariant; and the rooting of the three phases in
`exists_cube_root_one_iff_disc_neg`, which makes the third of a turn forced by the face rather
than assumed.

**Not derived here.** That the measured quotient is `2/3` is an input, carried as interval
hypotheses in `koide_lands`. Sumino, arXiv:0812.2103, assumes the condition on the vacuum and
proves a gauge correction cancels the electromagnetic one that would spoil it, which is
stability, not derivation, and his paper says so. Bhatt, Kaushik, Raj and Singh,
arXiv:2108.05787, equation (63), get exactly `2/3` from the roots of a characteristic cubic of
the exceptional Jordan algebra, as an observation. `theory/DEAD_ENDS.md` records the three routes
tried here and why none closed.

## Root

Instantiates `Trichotomy.flow` at `σ = −1` and rides
`Trichotomy.exists_cube_root_one_iff_disc_neg` for why the face has to be elliptic. The edges:

* `third_turn_is_the_cube_root` exhibits `flow (-1) (2π/3)` as the witness that theorem produces,
  matching its forced first slot `−1/2`, so the phase spacing is the algebra's and not a
  convention.
* `trinity_cos_sum` and `trinity_cos_sq_sum` are the first and second moments of that element's
  orbit, and `koide_of_trinity` is the quotient computed from them.
* `koide_lands` carries the measured masses as interval hypotheses on the square roots and
  concludes a bound, so the empirical step is in the type.

Everything here is on the elliptic face, because that is the only face with an element of
order three to carry three generations.
-/

namespace Apeiron
namespace Physics

open Trichotomy Real

/-- A third of a turn takes the first slot to minus a half.

    cos                        [how far along a turn goes across]
    Real.cos (2 * π / 3)       [the first slot after a third of a turn]

    So: the value `exists_cube_root_one_iff_disc_neg` forces on any cube root of the unit. -/
theorem cos_two_pi_div_three : Real.cos (2 * π / 3) = -(1 / 2) := by
  rw [show 2 * π / 3 = π - π / 3 by ring, Real.cos_sub, Real.cos_pi, Real.sin_pi,
    Real.cos_pi_div_three, Real.sin_pi_div_three]
  ring

/-- A third of a turn takes the second slot to half the square root of three.

    sin                        [how far along a turn goes up]
    Real.sin (2 * π / 3)       [the second slot after a third of a turn]
    Real.sqrt                  [the number that squares to give this] -/
theorem sin_two_pi_div_three : Real.sin (2 * π / 3) = Real.sqrt 3 / 2 := by
  rw [show 2 * π / 3 = π - π / 3 by ring, Real.sin_sub, Real.sin_pi, Real.cos_pi,
    Real.cos_pi_div_three, Real.sin_pi_div_three]
  ring

/-- Two thirds of a turn takes the first slot to minus a half as well.

    cos                        [how far along a turn goes across]
    Real.cos (4 * π / 3)       [the first slot after two thirds of a turn] -/
theorem cos_four_pi_div_three : Real.cos (4 * π / 3) = -(1 / 2) := by
  rw [show 4 * π / 3 = π + π / 3 by ring, Real.cos_add, Real.cos_pi, Real.sin_pi,
    Real.cos_pi_div_three, Real.sin_pi_div_three]
  ring

/-- Two thirds of a turn takes the second slot down.

    sin                        [how far along a turn goes up]
    Real.sin (4 * π / 3)       [the second slot after two thirds of a turn]
    Real.sqrt                  [the number that squares to give this] -/
theorem sin_four_pi_div_three : Real.sin (4 * π / 3) = -(Real.sqrt 3 / 2) := by
  rw [show 4 * π / 3 = π + π / 3 by ring, Real.sin_add, Real.sin_pi, Real.cos_pi,
    Real.cos_pi_div_three, Real.sin_pi_div_three]
  ring

/-- **A third of a turn is the cube root of the unit the elliptic face forces.**

    flow                                                        [the size-one flow of the algebra]
    mul                                                         [multiply two points, the σ way]
    one                                                      [the point multiplication leaves alone]
    ∧                                                           [both are true]
    ≠                                                           [is not equal to]
    flow (-1) (2 * π / 3)                                       [the point a third of a turn along]

    So: the witness `exists_cube_root_one_iff_disc_neg` produces, exhibited. Its first slot is
    `−1/2`, which that theorem shows is forced on every cube root of the unit other than the unit,
    so three generations spaced by a third of a turn is the algebra's own spacing. -/
theorem third_turn_is_the_cube_root :
    mul (-1) (mul (-1) (flow (-1) (2 * π / 3)) (flow (-1) (2 * π / 3))) (flow (-1) (2 * π / 3))
        = one
      ∧ flow (-1) (2 * π / 3) ≠ one
      ∧ (flow (-1) (2 * π / 3)).1 = -(1 / 2) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [flow_add, flow_add, show 2 * π / 3 + 2 * π / 3 + 2 * π / 3 = 2 * π by ring,
      flow_neg_one]
    simp [one, Real.cos_two_pi, Real.sin_two_pi]
  · intro h
    have hfst := congrArg Prod.fst h
    rw [flow_neg_one, cos_two_pi_div_three] at hfst
    norm_num [one] at hfst
  · rw [flow_neg_one, cos_two_pi_div_three]

/-- **The three phases cancel.**

    cos                                                              [how far a turn goes across]
    ℝ                                                                [the real numbers]

    So: the first moment of the order-three element's orbit vanishes, which is why a common size
    survives untouched in the sum of the three samples. -/
theorem trinity_cos_sum (δ : ℝ) :
    Real.cos δ + Real.cos (δ + 2 * π / 3) + Real.cos (δ + 4 * π / 3) = 0 := by
  rw [Real.cos_add, Real.cos_add, cos_two_pi_div_three, sin_two_pi_div_three,
    cos_four_pi_div_three, sin_four_pi_div_three]
  ring

/-- **The three phases carry three halves of the amplitude.**

    cos                                                              [how far a turn goes across]
    ℝ                                                                [the real numbers]

    So: the second moment of the orbit is `3/2`, which is the channel the amplitude enters the
    sum of squares through. -/
theorem trinity_cos_sq_sum (δ : ℝ) :
    Real.cos δ ^ 2 + Real.cos (δ + 2 * π / 3) ^ 2 + Real.cos (δ + 4 * π / 3) ^ 2 = 3 / 2 := by
  have pyth := Real.sin_sq_add_cos_sq δ
  have hr2 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  rw [Real.cos_add, Real.cos_add, cos_two_pi_div_three, sin_two_pi_div_three,
    cos_four_pi_div_three, sin_four_pi_div_three]
  linear_combination (3 / 2 : ℝ) * pyth + (Real.sin δ ^ 2 / 2) * hr2

/-- The quotient of three sizes by the square of their square roots added.

    ℝ                                                    [the real numbers]
    (x ^ 2 + y ^ 2 + z ^ 2) / (x + y + z) ^ 2            [the sizes added, over the sum squared]
    koideQuotient                                        [that quotient]

    So: the three arguments are the square roots of the sizes, so the numerator is the sizes
    added and the denominator is the square of their square roots added. -/
noncomputable def koideQuotient (x y z : ℝ) : ℝ := (x ^ 2 + y ^ 2 + z ^ 2) / (x + y + z) ^ 2

/-- How squarely three square roots point along the all-equal direction.

    ℝ                                                    [the real numbers]
    (x + y + z) ^ 2 / (3 * (x ^ 2 + y ^ 2 + z ^ 2))      [the squared cosine of that angle]
    coneAngleSq                                          [that squared cosine] -/
noncomputable def coneAngleSq (x y z : ℝ) : ℝ :=
  (x + y + z) ^ 2 / (3 * (x ^ 2 + y ^ 2 + z ^ 2))

/-- **The quotient is one third over the squared cone angle.**

    ℝ                                                    [the real numbers]
    ≠                                                    [is not equal to]
    coneAngleSq                                    [how squarely the roots point along the diagonal]
    koideQuotient                                        [the sizes added, over the sum squared]

    So: Foot's reading, arXiv:hep-ph/9402242 equations (2) and (3), as a theorem. The quotient is
    an angle in disguise, and `2/3` is the angle whose squared cosine is a half. -/
theorem cone_law (x y z : ℝ) (hsum : x + y + z ≠ 0) (hsq : x ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
    coneAngleSq x y z * koideQuotient x y z = 1 / 3 := by
  unfold coneAngleSq koideQuotient
  field_simp

/-- **The measured quotient is the quarter turn.**

    ℝ                                                    [the real numbers]
    ≠                                                    [is not equal to]
    coneAngleSq                                    [how squarely the roots point along the diagonal]
    koideQuotient                                        [the sizes added, over the sum squared]

    So: at `2/3` the vector of square roots stands at `45°` to the all-equal direction, half a
    right angle, and its squared cosine is a half. -/
theorem koide_is_quarter_turn (x y z : ℝ) (hsum : x + y + z ≠ 0)
    (hsq : x ^ 2 + y ^ 2 + z ^ 2 ≠ 0) (hkoide : koideQuotient x y z = 2 / 3) :
    coneAngleSq x y z = 1 / 2 := by
  have h := cone_law x y z hsum hsq
  rw [hkoide] at h
  linarith

/-- **One amplitude at three phases: the quotient is two plus its square, over six.**

    ℝ                                                     [the real numbers]
    ≠                                                     [is not equal to]
    koideQuotient                                         [the sizes added, over the sum squared]
    (2 + a ^ 2) / 6                                       [two plus the squared amplitude, over six]

    So: the size and the phase both cancel, and the whole quotient is the amplitude. Koide,
    arXiv:hep-ph/0005137 equation (2.5), and Brannen (2006) equation (7). -/
theorem koide_of_trinity (M a δ : ℝ) (hM : M ≠ 0) :
    koideQuotient (M * (1 + a * Real.cos δ)) (M * (1 + a * Real.cos (δ + 2 * π / 3)))
        (M * (1 + a * Real.cos (δ + 4 * π / 3))) = (2 + a ^ 2) / 6 := by
  have hz := trinity_cos_sum δ
  have hsq2 := trinity_cos_sq_sum δ
  unfold koideQuotient
  have hsum : M * (1 + a * Real.cos δ) + M * (1 + a * Real.cos (δ + 2 * π / 3))
      + M * (1 + a * Real.cos (δ + 4 * π / 3)) = 3 * M := by
    linear_combination (M * a) * hz
  have hsq : (M * (1 + a * Real.cos δ)) ^ 2 + (M * (1 + a * Real.cos (δ + 2 * π / 3))) ^ 2
      + (M * (1 + a * Real.cos (δ + 4 * π / 3))) ^ 2 = M ^ 2 * (3 + a ^ 2 * (3 / 2)) := by
    linear_combination (2 * a * M ^ 2) * hz + (a ^ 2 * M ^ 2) * hsq2
  rw [hsum, hsq]
  field_simp
  ring

/-- **The quotient is two thirds exactly when the squared amplitude is two.**

    ℝ                                                     [the real numbers]
    ≠                                                     [is not equal to]
    ↔                                                     [exactly when, true in both directions]
    koideQuotient                                         [the sizes added, over the sum squared]

    So: the measured value forces one amplitude and leaves the size and phase free. Brannen
    (2006) equation (8). -/
theorem koide_iff_amplitude_sq_two (M a δ : ℝ) (hM : M ≠ 0) :
    koideQuotient (M * (1 + a * Real.cos δ)) (M * (1 + a * Real.cos (δ + 2 * π / 3)))
        (M * (1 + a * Real.cos (δ + 4 * π / 3))) = 2 / 3
      ↔ a ^ 2 = 2 := by
  rw [koide_of_trinity M a δ hM, div_eq_div_iff (by norm_num) (by norm_num)]
  constructor <;> intro h <;> linarith

/-- **Reading the amplitude off any sector's own quotient.**

    ℝ                                                     [the real numbers]
    ≠                                                     [is not equal to]
    koideQuotient                                         [the sizes added, over the sum squared]

    So: six times the quotient less two is the squared amplitude, whatever the sector. The
    amplitude is not a free parameter per sector; it is the quotient. Koide,
    arXiv:hep-ph/0005137, Table 1. -/
theorem amplitude_sq_of_koide (M a δ : ℝ) (hM : M ≠ 0) :
    6 * koideQuotient (M * (1 + a * Real.cos δ)) (M * (1 + a * Real.cos (δ + 2 * π / 3)))
        (M * (1 + a * Real.cos (δ + 4 * π / 3))) - 2 = a ^ 2 := by
  rw [koide_of_trinity M a δ hM]
  ring

/-- **The three phases enter the product through one invariant.**

    ℝ                                                     [the real numbers]
    cos                                                   [how far a turn goes across]

    So: with the amplitude pinned, the only freedom left in the spectrum is `cos 3δ`. Unclaimed
    in the sweep recorded in `theory/LITERATURE.md` field F. -/
theorem azimuth_invariant (δ : ℝ) :
    Real.cos δ * Real.cos (δ + 2 * π / 3) * Real.cos (δ + 4 * π / 3)
      = (1 / 4) * Real.cos (3 * δ) := by
  have pyth := Real.sin_sq_add_cos_sq δ
  have hr2 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have h3 := Real.cos_three_mul δ
  rw [Real.cos_add, Real.cos_add, cos_two_pi_div_three, sin_two_pi_div_three,
    cos_four_pi_div_three, sin_four_pi_div_three, h3]
  linear_combination (-(Real.cos δ * Real.sin δ ^ 2) / 4) * hr2 + (-(3 * Real.cos δ) / 4) * pyth

/-- **All three samples are roots of one cubic.**

    ℝ                                                     [the real numbers]
    ℤ                                                     [the whole numbers, positive or negative]
    cos                                                   [how far a turn goes across]

    So: the three cosines are the three roots of `4c³ − 3c = cos 3δ`, so the spectrum is that one
    number, modulo the size and the amplitude. Unclaimed in the sweep. -/
theorem trinity_root (δ : ℝ) (k : ℤ) :
    4 * Real.cos (δ + 2 * π * (k : ℝ) / 3) ^ 3 - 3 * Real.cos (δ + 2 * π * (k : ℝ) / 3)
      = Real.cos (3 * δ) := by
  rw [← Real.cos_three_mul,
    show 3 * (δ + 2 * π * (k : ℝ) / 3) = 3 * δ + (k : ℝ) * (2 * π) from by ring,
    Real.cos_add_int_mul_two_pi]

/-- **The measured lepton masses put the quotient within three parts in a hundred thousand.**

    ℝ                                                     [the real numbers]
    ≤                                                     [is smaller than or equal to]
    ∧                                                     [both are true]
    <                                                     [is smaller than]
    koideQuotient                                         [the sizes added, over the sum squared]
    0.71484 ≤ x                                           [the electron's square root, in root MeV]
    10.27902 ≤ y                                          [the muon's square root]
    42.1513 ≤ z                                           [the tau's square root, PDG interval]

    So: the empirical step is in the type. The intervals bracket the square roots of the 2024
    particle data masses `0.51099895`, `105.6583755` and `1776.86 ± 0.12` MeV, and the conclusion
    is a bound on the distance from two thirds, by rational arithmetic alone. Nothing here
    derives the value; `theory/DEAD_ENDS.md` records what was tried. -/
theorem koide_lands (x y z : ℝ)
    (hx : 0.71484 ≤ x ∧ x ≤ 0.71485)
    (hy : 10.27902 ≤ y ∧ y ≤ 10.27903)
    (hz : 42.1513 ≤ z ∧ z ≤ 42.1543) :
    |koideQuotient x y z - 2 / 3| < 0.00003 := by
  obtain ⟨hx1, hx2⟩ := hx
  obtain ⟨hy1, hy2⟩ := hy
  obtain ⟨hz1, hz2⟩ := hz
  have hspos : (0 : ℝ) < x + y + z := by linarith
  have hden : (0 : ℝ) < 3 * (x + y + z) ^ 2 := by positivity
  have hkey : koideQuotient x y z - 2 / 3
      = (x ^ 2 + y ^ 2 + z ^ 2 - 4 * (x * y + y * z + z * x)) / (3 * (x + y + z) ^ 2) := by
    unfold koideQuotient
    field_simp
    ring
  rw [hkey, abs_div, abs_of_pos hden, div_lt_iff₀ hden]
  have hlow : (2824 : ℝ) ≤ (x + y + z) ^ 2 := by nlinarith
  have hnum : |x ^ 2 + y ^ 2 + z ^ 2 - 4 * (x * y + y * z + z * x)| < 0.248 := by
    rw [abs_lt]
    constructor <;> nlinarith
  nlinarith [hnum, hlow]

/-- **The intervals of `koide_lands` do bracket the measured square roots.**

    ∧                                                     [both are true]
    ≤                                                     [is smaller than or equal to]
    Real.sqrt                                             [the number that squares to give this]

    So: the hypotheses of `koide_lands` are satisfiable at the particle data values, so the bound
    is about the measured leptons and not about an empty interval. -/
theorem koide_intervals_bracket_the_data :
    (0.71484 ≤ Real.sqrt 0.51099895 ∧ Real.sqrt 0.51099895 ≤ 0.71485)
      ∧ (10.27902 ≤ Real.sqrt 105.6583755 ∧ Real.sqrt 105.6583755 ≤ 10.27903)
      ∧ (42.1513 ≤ Real.sqrt 1776.74 ∧ Real.sqrt 1776.98 ≤ 42.1543) := by
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ?_, ?_⟩
  · rw [show (0.71484 : ℝ) = Real.sqrt (0.71484 ^ 2) by
      rw [Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_le_sqrt (by norm_num)
  · rw [show (0.71485 : ℝ) = Real.sqrt (0.71485 ^ 2) by
      rw [Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_le_sqrt (by norm_num)
  · rw [show (10.27902 : ℝ) = Real.sqrt (10.27902 ^ 2) by
      rw [Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_le_sqrt (by norm_num)
  · rw [show (10.27903 : ℝ) = Real.sqrt (10.27903 ^ 2) by
      rw [Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_le_sqrt (by norm_num)
  · rw [show (42.1513 : ℝ) = Real.sqrt (42.1513 ^ 2) by
      rw [Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_le_sqrt (by norm_num)
  · rw [show (42.1543 : ℝ) = Real.sqrt (42.1543 ^ 2) by
      rw [Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_le_sqrt (by norm_num)

/-- The rank-three form whose points of size zero are the mass cone.

    ℝ                                                     [the real numbers]
    2 * (x + y + z) ^ 2 - 3 * (x ^ 2 + y ^ 2 + z ^ 2)     [twice the square of the sum, less three
                                                           times the sum of the squares]
    koideForm                                             [the rank-three form of the mass cone]

    So: on the all-equal direction it is positive and on the directions orthogonal to it negative,
    so it is indefinite of signature one and two, and its points of size zero are a genuine
    cone. -/
noncomputable def koideForm (x y z : ℝ) : ℝ := 2 * (x + y + z) ^ 2 - 3 * (x ^ 2 + y ^ 2 + z ^ 2)

/-- **The measured quotient says the square roots lie on a light cone.**

    ℝ                                                     [the real numbers]
    ≠                                                     [is not equal to]
    ↔                                                     [exactly when, true in both directions]
    koideQuotient                                         [the sizes added, over the sum squared]
    koideForm                                             [the rank-three form of the mass cone]

    So: the relation is not a numerical coincidence about a ratio, it is the statement that one
    vector has size zero for one indefinite form. That is the same kind of object
    `hasNullPair_iff_disc_pos` classifies in rank two, one rank up. -/
theorem koide_iff_null (x y z : ℝ) (hsum : x + y + z ≠ 0) :
    koideQuotient x y z = 2 / 3 ↔ koideForm x y z = 0 := by
  have hne : (x + y + z) ^ 2 ≠ 0 := pow_ne_zero 2 hsum
  unfold koideQuotient koideForm
  rw [div_eq_div_iff hne (by norm_num : (3 : ℝ) ≠ 0)]
  constructor <;> intro h <;> linarith

/-- **That form is indefinite, so its cone is neither empty nor everything.**

    ∃                                                     [there is at least one]
    ∧                                                     [both are true]
    <                                                     [is smaller than]
    ℝ                                                   [the real numbers: every number on the line]
    ×                                                     [paired with, two numbers side by side]
    koideForm                                             [the rank-three form of the mass cone]

    So: positive along the all-equal direction and negative across it, which is why the condition
    `koideForm = 0` picks out a cone rather than a point or the whole space. -/
theorem koideForm_indefinite :
    ∃ u v : ℝ × ℝ × ℝ, 0 < koideForm u.1 u.2.1 u.2.2 ∧ koideForm v.1 v.2.1 v.2.2 < 0 := by
  refine ⟨(1, 1, 1), (1, -1, 0), ?_, ?_⟩ <;> norm_num [koideForm]

/-- **The measured quotient equates the spread of the square roots with their mean squared,
    tripled.**

    ℝ                                                     [the real numbers]
    ≠                                                     [is not equal to]
    ↔                                                     [exactly when, true in both directions]
    koideQuotient                                         [the sizes added, over the sum squared]

    So: split the vector of square roots into its all-equal part and the rest. Two thirds says the
    two parts carry the same total weight. That is what any derivation of the value would have to
    supply, and it is a structural condition rather than a number. It is one line from Koide's own
    quotient formula, arXiv:hep-ph/0005137 equation (2.5), so it is a sharper reading and not a
    new result. -/
theorem koide_iff_spread (x y z : ℝ) (hsum : x + y + z ≠ 0) :
    koideQuotient x y z = 2 / 3
      ↔ (x - (x + y + z) / 3) ^ 2 + (y - (x + y + z) / 3) ^ 2 + (z - (x + y + z) / 3) ^ 2
          = 3 * ((x + y + z) / 3) ^ 2 := by
  rw [koide_iff_null x y z hsum]
  unfold koideForm
  constructor
  · intro h; linear_combination (-1 / 3 : ℝ) * h
  · intro h; linear_combination (-3 : ℝ) * h

end Physics
end Apeiron
