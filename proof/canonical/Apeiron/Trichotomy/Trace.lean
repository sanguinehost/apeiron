import Apeiron.Trichotomy.Derived.Application
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# Feedback: the loop, the execution formula, and which faces let it close

@name Trace
@cite girard-1989
@cite aitken-1926
@cite abramsky-haghverdi-scott-2002
@cite joyal-street-verity-1996
@cite haghverdi-scott-2010
@cite palmieri-2026
@grade proved

Application in the calculus is not multiplication, and `theory/DEAD_ENDS.md` closes that route
for good. This file takes the other one. In Girard's Geometry of Interaction application is
**composition with feedback**: put two things side by side, wire one output back to one input,
and solve the resulting loop. Solving it is a sum, not a product:

```
   a  +  b·c  +  b·d·c  +  b·d·d·c  +  ...    one term per trip round the loop
```

Each term is a product along one path and the terms are added across paths. That is what the
owner's "almost purely additive" asks for, and the multiplication that remains is the
algebra's own, taken along a path.

## Root

| from | here |
|---|---|
| `isUnit_iff_normForm_ne_zero` | `loop_solvable_iff_off_cone` |
| `exists_normForm_zero_iff_disc_nonneg` | `loop_always_solvable_iff_disc_neg` |
| `normForm_mul`, `normForm_eq_zero_iff_of_neg` | `normForm_cpow`, `elliptic_never_terminates` |
| `upper_mul`, `halves_ext`, `halves_zero`, `halves_one` | `nilpotent_iff_disc_zero` |
| `exists_normForm_zero_iff_disc_nonneg`, `normForm_mul` | `divergent_loop_of_disc_pos` |
| `mul_recip_cancel`, `mul_distrib_sub` | `restPoint_fixed`, `affineIter_sub_restPoint` |
| `normForm_mul`, `normForm_cpow` | `normForm_second_difference` |
| `isUnit_iff_normForm_ne_zero`, `mul_recip_cancel` | `aitken_exact` |
| `AnyRing.ringAffineFold_apply` | `affineIter_eq_ringAffineFold` |
| `aitken_exact`, `restPoint_fixed` | `aitken_exact_of_prefix_fold`, `aitken_hyperbolic_witness` |
| `affineIter_eq_ringAffineFold`, `depth_two_trichotomy` | `parabolic_square_zero_prefix_fold` |
| `collapse_always_exact_iff_disc_neg` | `collapse_always_exact_iff_disc_neg_prefix_fold` |
| `collapse_always_exact_iff_disc_neg` | `positive_face_prefix_fold_failure` |
| `normForm_eq_zero_iff_of_neg` | `collapse_always_exact_iff_disc_neg` |
| `upper_cpow`, `halves_zero`, `exists_normForm_zero_iff_disc_nonneg` | `depth_two_trichotomy` |
| `normForm_cpow`, `cpow_ne_zero_of_off_cone` | `parabolic_failing_and_dying_loops` |

The loop closes exactly when `one − d` lies off the cone, which is
`isUnit_iff_normForm_ne_zero` with the loop's own block in it, and the sign of σ decides how often
that fails. `loop_always_solvable_iff_disc_neg` is the sharp form: every loop except the
identity closes on the elliptic face, and on no other. Above zero the cone is two lines and a whole
line of loops does not close; at zero it is one line; below zero it is the origin alone.

Then the sign of σ decides the regime of the sum. `nilpotent_iff_disc_zero` says a non-zero loop
whose powers vanish exists exactly at `σ = 0`, so the execution formula terminates at a finite
stage on the parabolic face and on no other. `elliptic_never_terminates` says the sum runs
forever below zero, and `divergent_loop_of_disc_pos` exhibits a loop above zero that neither
closes nor terminates.

**The general feedback algebra is free of σ, and that is stated rather than hidden.**
`feedback_tighten`, `feedback_slide`, `feedback_vanishing_unit`, `feedback_vanishing_pair` and
`feedback_yank` are proved over an arbitrary commutative ring and use no property of this
algebra, so by `theory/NOVELTY.md`'s A1 spectator test they are facts about commutative rings.
They are here because the trace axioms have to be checked before anything may be called a
trace, and because the corpus had no trace. What is about this algebra is everything in the
second half, where the sign of σ decides whether the trace exists and how the series behaves.

## Where the axioms come from, and what commutativity costs

Joyal, Street and Verity's trace axioms, in the form the nLab states them: tightening,
`tr((g ⊗ id) ∘ f ∘ (h ⊗ id)) = g ∘ tr(f) ∘ h`; sliding, `tr((id ⊗ g) ∘ f) = tr(f ∘ (id ⊗ g))`;
vanishing, `tr¹(f) = f` and `tr^{X⊗Y}(f) = tr^X(tr^Y(f))`; strength, `tr(g ⊗ f) = g ⊗ tr(f)`;
and yanking, `tr(β) = id`. Here every block is one element of a commutative ring, so sliding is
`mul_comm` and tightening is a rearrangement. Vanishing over a pair is the one that earns its
keep: it is the Schur complement done twice against the Schur complement done once, and
`det_inverse` is why the two loops' inverses agree, `u · v` inverting the determinant of the
whole two-wire block.

## The predecessor's `B/(1 − A)`, at true strength

`../sanguine-ref/Substrate/StationaryPhase/SpinorLoop.lean`'s `constantStep_fixed_point` proved
that the affine step `x ↦ A·x + B` rests at `B/(1 − A)`, and read that as a `K`-realiser in the
bulk. The reading was culled; the equation was right and it is Girard's execution formula with
one wire. `feedback 0 B 1 u` with `u` inverting `1 − A` is `B/(1 − A)`, and
`execution_formula` says what the closed form is a closed form of: the sum over the number of
trips round the loop. Girard's own statement of the expansion is
`U(1 − σU)⁻¹ = U + UσU + UσUσU + ...`, whose length is the order of nilpotency of `σU`.
`execution_formula` is that sentence, with the order of nilpotency supplied as `n`.

## Depth two: three values in place of the loop

@cite aitken-1926

Aitken's 1926 extrapolation takes three consecutive values of a sequence and returns the limit
of the single geometric mode they sit on. A constant affine loop's distance to its resting point
is one such mode, `affineIter_sub_restPoint`, so the extrapolation is an identity rather than an
approximation, and the loop past its third value is optional. `../sanguine-ref`'s
`Substrate/StationaryPhase/SpinorLoop.lean` proved that over the reals, under `A ≠ 0`, `A ≠ 1`
and a seed away from the fixed point. Those are three conditions of one kind, because over a
field a non-zero element has non-zero size.

Here they come apart, and that is what the port is for. `normForm_second_difference` says the
size of the quantity the formula divides by is
`normForm σ (one − d)² · normForm σ d ^ n · normForm σ (s − rest)`, so by `normForm_mul` the
collapse is available exactly when all three factors lie off the null cone.
`collapse_always_exact_iff_disc_neg` sharpens that: the collapse holds for every loop that
closes exactly below zero, which is `loop_always_solvable_iff_disc_neg` one level up. That
theorem says which faces let a loop have an answer; this one says which faces let three of its
values compute that answer. `depth_two_trichotomy` names all three faces at once. At zero a loop
weight of size zero squares to nothing, so where the weight is what blocks the collapse,
`execution_formula` answers in two trips instead and depth two survives by the other mechanism.
Above zero the blocking weight closes and never dies out, so neither route is open. A seed of
size zero blocks the collapse at both of those faces as well, and that case has no fallback: the
loop is not nilpotent and the sum has no last term.

## What this gives the trace, and what it does not

`resolvent_from_three_terms` is stated over any commutative ring and is about `loopSum` itself,
so it resolves the execution formula's own partial sums: three of them determine the resolvent
`feedback` takes as an argument, with no nilpotency and no bound on the depth.
`resolvent_from_three_partial_sums` is the same statement on this carrier, where the third side
condition is discharged rather than assumed, because the seed is the origin and the initial gap
is the resolvent, whose size `normForm_recip` says is never zero. That is the bridge, and it
reaches the block presentation of the trace, the one `feedback` and `execution_formula` are
written in.

It does not reach the value of a route at a position. `Combinator/Interaction.lean`'s `Traces`
is a union of partial maps, and `traces_functional` says at most one of them is defined at any
position, so a route's trace there is one path's weight and not a sum over paths. There is no
sequence of partial sums to extrapolate, and `Route` carries a join but no subtraction, so the
second difference the collapse divides by cannot even be written on it. `theory/DEAD_ENDS.md`
records the three attempts and the primitive that is missing, which is an additive structure on
routes of the kind Haghverdi and Scott's partially additive categories carry.
-/

noncomputable section

namespace Apeiron
namespace Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- THE FEEDBACK OF A BLOCK, OVER ANY COMMUTATIVE RING
-- ═══════════════════════════════════════════════════════════════════

section AnyCommRing

variable {R : Type*} [CommRing R]

/-- The feedback of a two-by-two block, with the loop's inverse supplied.

@cite girard-1989

    {R : Type*} [CommRing R]           [any commutative ring]
    (a b c u : R)                      [direct path, two loop legs, loop inverse]
    a + b * u * c                      [go straight, or out and round and back]
    : R                                [one element]

    So: the block is `a` from input to output, `c` from input into the loop, `b` from the loop
    back to the output, and `d` around the loop, with `u` inverting `1 − d`. The inverse is an
    argument rather than a field so that every theorem below says which loop it needs solved. -/
def feedback (a b c u : R) : R := a + b * u * c

/-- Tightening: pre-composing and post-composing pass through the loop.

@cite joyal-street-verity-1996

    {R : Type*} [CommRing R]                                      [any commutative ring]
    (g h a b c u : R)                                             [two maps and a block]
    feedback (g * a * h) (g * b) (c * h) u                        [wrap the block, then loop]
    = g * feedback a b c u * h                                    [or loop, then wrap]

    So: Joyal, Street and Verity's naturality in the two untraced wires. No hypothesis on the
    loop is needed: whatever `u` is, it commutes past `g` and `h`. -/
theorem feedback_tighten (g h a b c u : R) :
    feedback (g * a * h) (g * b) (c * h) u = g * feedback a b c u * h := by
  simp only [feedback]
  ring

/-- Sliding: a map on the loop wire may be pushed to either end.

@cite joyal-street-verity-1996

    {R : Type*} [CommRing R]                                      [any commutative ring]
    (a b c g u : R)                                               [a block and a loop map]
    feedback a b (g * c) u = feedback a (b * g) c u               [either end gives the same]

    So: dinaturality in the traced wire. In this setting it is `mul_comm` and nothing more,
    which is worth saying plainly: a commutative carrier makes the hardest trace axiom free and
    the reader should not mistake that for evidence the construction is deep. -/
theorem feedback_slide (a b c g u : R) :
    feedback a b (g * c) u = feedback a (b * g) c u := by
  simp only [feedback]
  ring

/-- Vanishing over the unit: a loop with no leg is no loop.

@cite joyal-street-verity-1996

    {R : Type*} [CommRing R]                        [any commutative ring]
    (a b c u : R)                                   [a block and a loop inverse]
    feedback a 0 c u = a                            [no way back from the loop]
    feedback a b 0 u = a                            [or no way into it]
    ∧                                               [both together]

    So: `tr¹(f) = f`. Tracing over a wire that carries nothing returns the direct path. -/
theorem feedback_vanishing_unit (a b c u : R) :
    feedback a 0 c u = a ∧ feedback a b 0 u = a := by
  constructor <;> simp [feedback]

/-- Yanking: feeding the loop straight back gives the identity.

@cite joyal-street-verity-1996

    {R : Type*} [CommRing R]              [any commutative ring]
    feedback 0 1 1 1 = (1 : R)            [no direct path, both legs open, no loop map]

    So: `tr(β) = id`. The block with `a = 0`, `b = c = 1` and `d = 0` is the swap, its loop
    inverse is `1`, and one trip round returns the input. -/
theorem feedback_yank : feedback (0 : R) 1 1 1 = 1 := by
  simp [feedback]

/-- The two loops' inverses multiply to the whole block's.

@cite joyal-street-verity-1996

    {R : Type*} [CommRing R]                                     [any commutative ring]
    (d₁₁ d₁₂ d₂₁ d₂₂ u v : R)                                    [a two-by-two loop block]
    (hu : (1 - d₂₂) * u = 1)                                     [the second wire's loop closes]
    (hv : (1 - (d₁₁ + d₁₂ * u * d₂₁)) * v = 1)                   [and the first, after that]
    ((1 - d₁₁) * (1 - d₂₂) - d₁₂ * d₂₁) * (u * v) = 1            [so the whole block closes]

    So: the determinant of `1 − D` is `(1 − d₂₂)` times the Schur complement, so its inverse is
    the product of the two inverses. This is what makes tracing two wires at once agree with
    tracing them one after the other. -/
theorem det_inverse (d₁₁ d₁₂ d₂₁ d₂₂ u v : R)
    (hu : (1 - d₂₂) * u = 1)
    (hv : (1 - (d₁₁ + d₁₂ * u * d₂₁)) * v = 1) :
    ((1 - d₁₁) * (1 - d₂₂) - d₁₂ * d₂₁) * (u * v) = 1 := by
  linear_combination (v * (1 - d₁₁)) * hu + hv

/-- Vanishing over a pair: tracing two wires at once is tracing them one at a time.

@cite joyal-street-verity-1996
@cite haghverdi-scott-2010

    {R : Type*} [CommRing R]                                     [any commutative ring]
    (a b₁ b₂ c₁ c₂ d₁₁ d₁₂ d₂₁ d₂₂ u v : R)                      [a block with two loop wires]
    (hu : (1 - d₂₂) * u = 1)                                     [the second wire's loop closes]
    (hv : (1 - (d₁₁ + d₁₂ * u * d₂₁)) * v = 1)                   [and the first, after that]
    feedback (a + b₂ * u * c₂) (b₁ + b₂ * u * d₂₁)              [the second wire, then the first]
      (c₁ + d₁₂ * u * c₂) v                                      [is the iterated trace]
    = a + (b₁ * ((1 - d₂₂) * c₁ + d₁₂ * c₂)                      [and equals both at once]
      + b₂ * (d₂₁ * c₁ + (1 - d₁₁) * c₂)) * (u * v)              [against the adjugate]

    So: `tr^{X⊗Y}(f) = tr^X(tr^Y(f))`, the axiom that makes this a trace rather than a formula.
    The right-hand side is the adjugate of `1 − D` over its determinant, and `det_inverse` says
    `u · v` is that determinant's inverse. -/
theorem feedback_vanishing_pair (a b₁ b₂ c₁ c₂ d₁₁ d₁₂ d₂₁ d₂₂ u v : R)
    (hu : (1 - d₂₂) * u = 1)
    (hv : (1 - (d₁₁ + d₁₂ * u * d₂₁)) * v = 1) :
    feedback (a + b₂ * u * c₂) (b₁ + b₂ * u * d₂₁) (c₁ + d₁₂ * u * c₂) v
      = a + (b₁ * ((1 - d₂₂) * c₁ + d₁₂ * c₂)
        + b₂ * (d₂₁ * c₁ + (1 - d₁₁) * c₂)) * (u * v) := by
  simp only [feedback]
  linear_combination (-(v * b₁ * c₁)) * hu + (-(b₂ * u * c₂)) * hv

-- ═══════════════════════════════════════════════════════════════════
-- THE EXECUTION FORMULA
-- ═══════════════════════════════════════════════════════════════════

/-- The sum over the number of trips round the loop, to a given depth.

@cite girard-1989

    {R : Type*} [CommRing R]                  [any commutative ring]
    (d : R)                                   [what one trip round the loop does]
    ℕ → R                                     [how many trips are counted]

    So: `0` trips is the empty sum, and each extra trip adds one at the front and sends the
    rest once more round. This is Horner's form of `1 + d + d² + ...`. -/
def loopSum (d : R) : ℕ → R
  | 0 => 0
  | n + 1 => 1 + d * loopSum d n

/-- Counting trips is summing powers.

    {R : Type*} [CommRing R]                                     [any commutative ring]
    (d : R) (n : ℕ)                                              [a loop and a depth]
    loopSum d n = ∑ k ∈ Finset.range n, d ^ k                    [one term per trip]

    So: the additive reading, written out. The loop contributes a sum across paths, and each
    path contributes a product along it. -/
theorem loopSum_eq_sum (d : R) (n : ℕ) :
    loopSum d n = ∑ k ∈ Finset.range n, d ^ k := by
  induction n with
  | zero => simp [loopSum]
  | succ n ih =>
    have hs : ∑ k ∈ Finset.range (n + 1), d ^ k
        = (∑ k ∈ Finset.range n, d ^ (k + 1)) + d ^ 0 :=
      Finset.sum_range_succ' (fun k => d ^ k) n
    rw [loopSum, ih, hs, pow_zero, Finset.mul_sum,
      add_comm (∑ k ∈ Finset.range n, d ^ (k + 1)) 1]
    congr 1
    exact Finset.sum_congr rfl fun k _ => by rw [pow_succ]; ring

/-- The sum telescopes against the loop.

    {R : Type*} [CommRing R]                                     [any commutative ring]
    (d : R) (n : ℕ)                                              [a loop and a depth]
    (1 - d) * loopSum d n = 1 - d ^ n                            [the middle terms cancel]

    So: everything but the first and last term cancels, which is why the sum is an inverse as
    soon as the last term is zero. -/
theorem loopSum_telescopes (d : R) (n : ℕ) :
    (1 - d) * loopSum d n = 1 - d ^ n := by
  induction n with
  | zero => simp [loopSum]
  | succ n ih =>
    simp only [loopSum, pow_succ]
    linear_combination d * ih

/-- **The execution formula.** A loop that dies out is a finite sum over paths.

@cite girard-1989
@cite abramsky-haghverdi-scott-2002

    {R : Type*} [CommRing R]                                     [any commutative ring]
    (a b c d u : R) (n : ℕ)                                      [a block, an inverse, a depth]
    (hnil : d ^ n = 0)                                           [the loop dies after n trips]
    (hu : (1 - d) * u = 1)                                       [and it closes]
    feedback a b c u = a + b * loopSum d n * c                   [the feedback is the finite sum]

    So: Girard's `U(1 − σU)⁻¹ = U + UσU + UσUσU + ...`, whose length he says is the order of
    nilpotency of `σU`. Here `n` is that order and the sum is exact rather than a limit: no
    convergence is used and no topology is needed. Below, `nilpotent_iff_disc_zero` says which
    face of the trichotomy supplies a loop that dies out. -/
theorem execution_formula (a b c d u : R) (n : ℕ) (hnil : d ^ n = 0)
    (hu : (1 - d) * u = 1) :
    feedback a b c u = a + b * loopSum d n * c := by
  have ht : (1 - d) * loopSum d n = 1 := by
    rw [loopSum_telescopes, hnil, sub_zero]
  have huv : u = loopSum d n := by
    calc u = u * ((1 - d) * loopSum d n) := by rw [ht, mul_one]
      _ = ((1 - d) * u) * loopSum d n := by ring
      _ = loopSum d n := by rw [hu, _root_.one_mul]
  rw [feedback, huv]

/-- The gap between consecutive partial sums is one trip's weight.

@cite girard-1989

    {R : Type*} [CommRing R]                                     [any commutative ring]
    (d : R) (n : ℕ)                                              [a loop and a depth]
    loopSum d (n + 1) - loopSum d n = d ^ n                      [one more trip adds one term]

    So: `loopSum_telescopes` read as a difference. The partial sums of the execution formula
    move by the term the next path contributes, which is what makes the sequence of partial
    sums an affine recurrence in the loop weight rather than an arbitrary sequence. -/
theorem loopSum_step (d : R) (n : ℕ) : loopSum d (n + 1) - loopSum d n = d ^ n := by
  have h := loopSum_telescopes d n
  simp only [loopSum]
  linear_combination -h

/-- The second difference of the partial sums, in closed form.

    {R : Type*} [CommRing R]                                     [any commutative ring]
    (d : R) (n : ℕ)                                              [a loop and a depth]
    loopSum d (n + 2) - 2 * loopSum d (n + 1)                  [the second difference of]
      + loopSum d n = d ^ (n + 1) - d ^ n                       [the partial sums, in closed form]

    So: `loopSum_step` twice. This element is the whole side condition of the collapse below:
    three partial sums determine the resolvent wherever this one element can be divided by. -/
theorem loopSum_second_difference (d : R) (n : ℕ) :
    loopSum d (n + 2) - 2 * loopSum d (n + 1) + loopSum d n = d ^ (n + 1) - d ^ n := by
  have h0 := loopSum_step d n
  have h1 := loopSum_step d (n + 1)
  linear_combination h1 - h0

/-- **Three partial sums of the execution formula determine the resolvent.**

@cite aitken-1926
@cite girard-1989

    {R : Type*} [CommRing R]                                     [any commutative ring]
    (d u v : R) (n : ℕ)                                          [a loop, two inverses, a depth]
    (hu : (1 - d) * u = 1)                                       [the loop closes]
    (hv : (loopSum d (n + 2) - 2 * loopSum d (n + 1)           [the second difference]
      + loopSum d n) * v = 1)                                   [closes as well]
    loopSum d n - (loopSum d (n + 1) - loopSum d n) ^ 2 * v = u  [the extrapolation is the
                                                                  resolvent]

    So: Aitken's 1926 extrapolation, applied to the partial sums `execution_formula` is a
    statement about, returns the resolvent that formula needs. It is an identity at every
    depth, not a limit: no convergence is used and no nilpotency is assumed, so the sum past
    its third term is optional wherever the second difference can be divided by. Both inverses
    are arguments, the way `execution_formula` takes `u` as one, so the statement carries the
    divisions it needs rather than hiding them in a quotient. -/
theorem resolvent_from_three_terms (d u v : R) (n : ℕ) (hu : (1 - d) * u = 1)
    (hv : (loopSum d (n + 2) - 2 * loopSum d (n + 1) + loopSum d n) * v = 1) :
    loopSum d n - (loopSum d (n + 1) - loopSum d n) ^ 2 * v = u := by
  rw [loopSum_second_difference] at hv
  rw [loopSum_step]
  have hs := loopSum_telescopes d n
  have hpow : d ^ (n + 1) = d * d ^ n := by rw [pow_succ]; ring
  rw [hpow] at hv
  have hkey : d ^ n * u = -((d ^ n) ^ 2 * v) := by
    calc d ^ n * u = d ^ n * u * ((d * d ^ n - d ^ n) * v) := by rw [hv]; ring
      _ = -((d ^ n) ^ 2 * v) * ((1 - d) * u) := by ring
      _ = -((d ^ n) ^ 2 * v) := by rw [hu]; ring
  have hsn : loopSum d n = u * (1 - d ^ n) := by
    calc loopSum d n = loopSum d n * ((1 - d) * u) := by rw [hu]; ring
      _ = (1 - d) * loopSum d n * u := by ring
      _ = (1 - d ^ n) * u := by rw [hs]
      _ = u * (1 - d ^ n) := by ring
  rw [hsn]
  linear_combination -hkey

end AnyCommRing

-- ═══════════════════════════════════════════════════════════════════
-- WHICH LOOPS CLOSE, AND ON WHICH FACE
-- ═══════════════════════════════════════════════════════════════════

/-- The loop closes exactly when its block lies off the cone.

    (σ : ℝ)                                                 [the parameter]
    (d : Cell)                                              [what one trip round the loop does]
    (∃ u : Cell, mul σ (one - d) u = one)                   [the loop has a solution]
    ↔                                                       [exactly when]
    normForm σ (one - d) ≠ 0                                [one minus it lies off the cone]

    So: solving `x = d·x + c` is inverting `one − d`, and `isUnit_iff_normForm_ne_zero` says
    inverting is exactly lying off the cone. The cone is where feedback fails, and σ
    decides how big the cone is. -/
theorem loop_solvable_iff_off_cone (σ : ℝ) (d : Cell) :
    (∃ u : Cell, mul σ (one - d) u = one) ↔ normForm σ (one - d) ≠ 0 :=
  isUnit_iff_normForm_ne_zero σ (one - d)

/-- **Every loop but the identity closes exactly on the elliptic face.**

    (σ : ℝ)                                                      [the parameter]
    (∀ d : Cell, d ≠ one → ∃ u : Cell, mul σ (one - d) u = one)  [every loop but one closes]
    ↔                                                            [exactly when]
    σ < 0                                                        [σ is below zero]

    So: below zero the only point of size zero is the origin, so the only unsolvable loop is
    `d = one`, which is the loop that never leaves. At zero and above the cone is a line or two
    lines, and a whole line of loops fails to close. This is the trichotomy answering a
    question about feedback rather than about algebra: which faces let a computation that feeds
    its output back to its input have an answer. -/
theorem loop_always_solvable_iff_disc_neg (σ : ℝ) :
    (∀ d : Cell, d ≠ one → ∃ u : Cell, mul σ (one - d) u = one) ↔ σ < 0 := by
  constructor
  · intro h
    by_contra hσ
    obtain ⟨ν, hν0, hνn⟩ := (exists_normForm_zero_iff_disc_nonneg σ).mpr (not_lt.mp hσ)
    obtain ⟨u, hu⟩ := h (one - ν) fun hcon => hν0 (sub_eq_self.mp hcon)
    rw [sub_sub_cancel] at hu
    exact (isUnit_iff_normForm_ne_zero σ ν).mp ⟨u, hu⟩ hνn
  · intro hσ d hd
    refine (isUnit_iff_normForm_ne_zero σ (one - d)).mpr fun hn => hd ?_
    exact (sub_eq_zero.mp ((normForm_eq_zero_iff_of_neg hσ (one - d)).mp hn)).symm

-- ═══════════════════════════════════════════════════════════════════
-- HOW THE LOOP'S POWERS BEHAVE, FACE BY FACE
-- ═══════════════════════════════════════════════════════════════════

/-- What n trips round the loop do.

    (σ : ℝ)                        [the parameter]
    (d : Cell)                     [what one trip does]
    ℕ → Cell                       [what n trips do]

    So: no trips is the unit, and one more trip multiplies by `d`. -/
def cpow (σ : ℝ) (d : Cell) : ℕ → Cell
  | 0 => one
  | n + 1 => mul σ d (cpow σ d n)

/-- The size of n trips is the size of one trip, n times over.

    (σ : ℝ) (d : Cell) (n : ℕ)                                [the parameter, a loop, a depth]
    normForm σ (cpow σ d n) = normForm σ d ^ n                   [sizes multiply along the loop]

    So: `normForm_mul` carried along the loop. Everything below reads off this one equation. -/
theorem normForm_cpow (σ : ℝ) (d : Cell) (n : ℕ) :
    normForm σ (cpow σ d n) = normForm σ d ^ n := by
  induction n with
  | zero => simp [cpow, normForm, one]
  | succ n ih => rw [cpow, normForm_mul, ih, pow_succ]; ring

/-- The first channel of n trips is that channel of one trip, n times over.

    {σ : ℝ} (hσ : 0 ≤ σ)                                    [the parameter, at zero or above]
    (d : Cell) (n : ℕ)                                           [a loop and a depth]
    upper σ (cpow σ d n) = upper σ d ^ n                         [the channel multiplies] -/
theorem upper_cpow {σ : ℝ} (hσ : 0 ≤ σ) (d : Cell) (n : ℕ) :
    upper σ (cpow σ d n) = upper σ d ^ n := by
  induction n with
  | zero => simp [cpow, (halves_one σ).1]
  | succ n ih => rw [cpow, upper_mul hσ, ih, pow_succ]; ring

/-- The second channel of n trips is that channel of one trip, n times over.

    {σ : ℝ} (hσ : 0 ≤ σ)                                    [the parameter, at zero or above]
    (d : Cell) (n : ℕ)                                           [a loop and a depth]
    lower σ (cpow σ d n) = lower σ d ^ n                         [the channel multiplies] -/
theorem lower_cpow {σ : ℝ} (hσ : 0 ≤ σ) (d : Cell) (n : ℕ) :
    lower σ (cpow σ d n) = lower σ d ^ n := by
  induction n with
  | zero => simp [cpow, (halves_one σ).2]
  | succ n ih => rw [cpow, lower_mul hσ, ih, pow_succ]; ring

/-- A loop off the cone never dies out.

    {σ : ℝ} {d : Cell}                             [the parameter and a loop]
    (h : normForm σ d ≠ 0)                              [the loop lies off the cone]
    (n : ℕ)                                             [any depth]
    cpow σ d n ≠ 0                                      [n trips never come to nothing]

    So: sizes multiply, so a non-zero size stays non-zero, and the origin has size zero. -/
theorem cpow_ne_zero_of_off_cone {σ : ℝ} {d : Cell} (h : normForm σ d ≠ 0) (n : ℕ) :
    cpow σ d n ≠ 0 := by
  intro hc
  have hn := normForm_cpow σ d n
  rw [hc] at hn
  have hz : normForm σ (0 : Cell) = 0 := by simp [normForm]
  rw [hz] at hn
  exact pow_ne_zero n h hn.symm

/-- **A loop that dies out exists exactly at the degenerate setting.**

    (σ : ℝ)                                                      [the parameter]
    (∃ (d : Cell) (n : ℕ), d ≠ 0 ∧ cpow σ d n = 0)               [some non-trivial loop dies out]
    ↔                                                            [exactly when]
    σ = 0                                                        [σ is at zero]

    So: the execution formula terminates at a finite stage on the parabolic face and on no
    other. Below zero nothing but the origin has size zero. Above zero the cone is two lines,
    but a point on one of them has a non-zero reading on the other channel, and that reading is
    raised to the power `n` rather than killed. Only at zero do the two channels coincide, and
    there the generator squares to nothing. Girard's existence condition for the execution
    formula is nilpotency; this says which face of the trichotomy supplies it. -/
theorem nilpotent_iff_disc_zero (σ : ℝ) :
    (∃ (d : Cell) (n : ℕ), d ≠ 0 ∧ cpow σ d n = 0) ↔ σ = 0 := by
  have hpow : ∀ (x : ℝ) (m : ℕ), x ^ m = 0 → x = 0 := by
    intro x m hxm
    by_contra hx0
    exact pow_ne_zero m hx0 hxm
  have hz0 : normForm σ (0 : Cell) = 0 := by simp [normForm]
  constructor
  · rintro ⟨d, n, hd0, hdn⟩
    have hz : normForm σ d = 0 := by
      have h := normForm_cpow σ d n
      rw [hdn, hz0] at h
      exact hpow _ n h.symm
    rcases eq_or_lt_of_le ((exists_normForm_zero_iff_disc_nonneg σ).mp ⟨d, hd0, hz⟩) with h | hpos
    · exact h.symm
    · exfalso
      have hu : upper σ d = 0 := by
        refine hpow _ n ?_
        rw [← upper_cpow hpos.le, hdn, (halves_zero σ).1]
      have hl : lower σ d = 0 := by
        refine hpow _ n ?_
        rw [← lower_cpow hpos.le, hdn, (halves_zero σ).2]
      exact hd0 (halves_ext hpos (by rw [hu, (halves_zero σ).1])
        (by rw [hl, (halves_zero σ).2]))
  · rintro rfl
    refine ⟨((0 : ℝ), (1 : ℝ)), 2, ?_, ?_⟩
    · intro h
      have h2 := congrArg Prod.snd h
      norm_num at h2
    · simp [cpow, mul, one, Prod.ext_iff]

/-- Below zero no loop but the trivial one ever dies out.

    {d : Cell} (hd : d ≠ 0)                             [a loop that is not the origin]
    (n : ℕ)                                             [any depth]
    cpow (-1) d n ≠ 0                                   [n trips never come to nothing]

    So: on the elliptic face the size is definite, so it vanishes only at the origin
    (`normForm_eq_zero_iff_of_neg`). The execution formula is an infinite sum there and its
    behaviour is set by whether the size is below, at, or above one. -/
theorem elliptic_never_terminates {d : Cell} (hd : d ≠ 0) (n : ℕ) :
    cpow (-1) d n ≠ 0 :=
  cpow_ne_zero_of_off_cone
    (fun h => hd ((normForm_eq_zero_iff_of_neg (by norm_num) d).mp h)) n

/-- A loop of unit size stays unit size for ever.

    {σ : ℝ} {d : Cell}                             [the parameter and a loop]
    (h : normForm σ d = 1)                              [one trip preserves size]
    (n : ℕ)                                             [any depth]
    normForm σ (cpow σ d n) = 1                         [so do n trips]

    So: the isometric regime. The sum neither dies out nor grows in size, whatever the face.
    `Physics/Quadratic.lean`'s `unitNorm_elliptic` puts the rotations here at `σ = −1`, and its
    `unitNorm_hyperbolic` puts the boosts here at `σ = 1`, which is why size alone does not
    separate the faces and the channel below does. -/
theorem unit_loop_isometric {σ : ℝ} {d : Cell} (h : normForm σ d = 1) (n : ℕ) :
    normForm σ (cpow σ d n) = 1 := by
  rw [normForm_cpow, h, one_pow]

/-- Above zero a boost keeps its size and its channel runs away.

    (t : ℝ) (n : ℕ)                                                   [a boost and a depth]
    upper 1 (cpow 1 (Real.cosh t, Real.sinh t) n) = Real.exp t ^ n    [geometric growth]

    So: the hyperbolic regime, and the reason size is the wrong measure of divergence. The
    boost has size one at every depth (`unitNorm_hyperbolic`), so the loop is isometric in the
    norm form, while one channel reads `eᵗ` per trip and the other reads `e⁻ᵗ`. A feedback sum
    over such a loop neither terminates nor settles. -/
theorem hyperbolic_loop_grows (t : ℝ) (n : ℕ) :
    upper 1 (cpow 1 (Real.cosh t, Real.sinh t) n) = Real.exp t ^ n := by
  rw [upper_cpow (by norm_num)]
  congr 1
  simp only [upper, Real.sqrt_one, _root_.one_mul]
  exact Real.cosh_add_sinh t

/-- **At zero a failing loop need not die, while every loop that dies closes.**

    normForm 0 (one - ((1, 1) : Cell)) = 0             [the loop (1,1) fails to close]
    ∀ n : ℕ, cpow 0 ((1, 1) : Cell) n ≠ 0              [and never dies out]
    ∀ (d : Cell) (n : ℕ), cpow 0 d n = 0 →             [while any loop that dies]
      d.1 = 0 ∧ normForm 0 (one - d) ≠ 0                [has first slot zero and closes]

    So: the failing and dying loops at the parabolic setting are not the same set. The loop
    `(1, 1)` witnesses failure without termination; a terminating loop has first slot zero, so
    `one - d` has norm form one and lies off the cone. -/
theorem parabolic_failing_and_dying_loops :
    normForm 0 (one - ((1, 1) : Cell)) = 0 ∧
      (∀ n : ℕ, cpow 0 ((1, 1) : Cell) n ≠ 0) ∧
      ∀ (d : Cell) (n : ℕ), cpow 0 d n = 0 →
        d.1 = 0 ∧ normForm 0 (one - d) ≠ 0 := by
  refine ⟨?_, ?_, ?_⟩
  · norm_num [normForm, one, AnyRing.ringNormForm, AnyRing.ringOne]
  · intro n
    exact cpow_ne_zero_of_off_cone
      (by norm_num [normForm, AnyRing.ringNormForm]) n
  · intro d n hdn
    have hpow : ∀ (x : ℝ) (m : ℕ), x ^ m = 0 → x = 0 := by
      intro x m hxm
      by_contra hx0
      exact pow_ne_zero m hx0 hxm
    have hdNorm : normForm 0 d = 0 := by
      have hn := normForm_cpow 0 d n
      rw [hdn] at hn
      have hz : normForm 0 (0 : Cell) = 0 := by simp [normForm]
      rw [hz] at hn
      exact hpow _ n hn.symm
    have hd1 : d.1 = 0 := by
      simp only [normForm, AnyRing.ringNormForm] at hdNorm
      nlinarith
    refine ⟨hd1, ?_⟩
    norm_num [normForm, one, AnyRing.ringNormForm, AnyRing.ringOne, hd1]

/-- **Above zero a loop can fail to close and still never die out.**

    {σ : ℝ} (hσ : 0 < σ)                               [the parameter, above zero]
    ∃ d : Cell                                              [there is a loop]
    d ≠ one                                                 [other than the one that never
    leaves]                                                 [so the case is not trivial]
    normForm σ (one - d) = 0                                [that does not close]
    ∀ n : ℕ, cpow σ d n ≠ 0                                 [and does not die out either]
    ∧                                                       [all three together]

    So: the third regime. On the split face the cone is two lines, so there are loops the
    feedback equation cannot solve, and those same loops have non-zero size, so the execution
    sum has no last term. Neither the closed form nor the series is available. Below zero this
    cannot happen, `loop_always_solvable_iff_disc_neg`. At zero `(1, 1)` fails to close and never
    dies, while every loop that dies has first slot zero and closes,
    `parabolic_failing_and_dying_loops`. -/
theorem divergent_loop_of_disc_pos {σ : ℝ} (hσ : 0 < σ) :
    ∃ d : Cell, d ≠ one ∧ normForm σ (one - d) = 0 ∧ ∀ n : ℕ, cpow σ d n ≠ 0 := by
  have hs2 : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt hσ.le
  have hinv : σ * (1 / Real.sqrt σ) ^ 2 = 1 := by
    rw [div_pow, one_pow, hs2, mul_one_div, div_self (ne_of_gt hσ)]
  refine ⟨((0 : ℝ), -(1 / Real.sqrt σ)), ?_, ?_, ?_⟩
  · intro hcon
    have h1 := congrArg Prod.fst hcon
    norm_num at h1
  · have hpt : one - ((0 : ℝ), -(1 / Real.sqrt σ)) = ((1 : ℝ), 1 / Real.sqrt σ) := by
      simp [one]
    rw [hpt]
    show (1 : ℝ) ^ 2 - σ * (1 / Real.sqrt σ) ^ 2 = 0
    linear_combination -hinv
  · refine cpow_ne_zero_of_off_cone ?_
    show (0 : ℝ) ^ 2 - σ * (-(1 / Real.sqrt σ)) ^ 2 ≠ 0
    intro hcon
    have hz : σ * (1 / Real.sqrt σ) ^ 2 = 0 := by linear_combination -hcon
    rw [hinv] at hz
    norm_num at hz

-- ═══════════════════════════════════════════════════════════════════
-- DEPTH TWO: THREE ITERATES IN PLACE OF THE LOOP
-- ═══════════════════════════════════════════════════════════════════

/-- The affine loop, one trip at a time.

    (σ : ℝ)                          [the parameter]
    (d c s : Cell)                   [a loop weight, an injection, a seed]
    ℕ → Cell                         [where the loop sits after that many trips]

    So: `S(t) = d·S(t−1) + c` on this carrier. This is the machine running, and it is the
    baseline the collapse below is measured against. -/
def affineIter (σ : ℝ) (d c s : Cell) : ℕ → Cell
  | 0 => s
  | n + 1 => mul σ d (affineIter σ d c s n) + c

/-- Where the affine loop comes to rest.

    (σ : ℝ)                                   [the parameter]
    (d c : Cell)                              [a loop weight and an injection]
    mul σ c (recip σ (one - d))               [the injection times the loop's resolvent]
    : Cell                                    [one point]

    So: `c` over `one − d`, the same resolvent `feedback` carries and the one
    `loop_solvable_iff_off_cone` says exists off the cone. On the cone `recip` returns the
    origin, which is why every statement below carries the hypothesis that `one − d` has
    non-zero size. -/
def restPoint (σ : ℝ) (d c : Cell) : Cell := mul σ c (recip σ (one - d))

/-- Aitken's delta-squared extrapolation of three points, saying nothing about where they
    came from.

@cite aitken-1926

    (σ : ℝ)                                                 [the parameter]
    (x₀ x₁ x₂ : Cell)                                       [three points]
    x₀ - mul σ (mul σ (x₁ - x₀) (x₁ - x₀))                  [the first, less the first]
      (recip σ (x₂ - (x₁ + x₁) + x₀))                       [difference squared over the second]
    : Cell                                                  [one point]

    So: Aitken 1926 transcribed into this algebra, with `mul σ` for the square and `recip` for
    the division. The doubling is written `x₁ + x₁` because Lean's numerals on a pair are
    componentwise and would not be this algebra's own. -/
def aitken (σ : ℝ) (x₀ x₁ x₂ : Cell) : Cell :=
  x₀ - mul σ (mul σ (x₁ - x₀) (x₁ - x₀)) (recip σ (x₂ - (x₁ + x₁) + x₀))

/-- The resting point rests.

    (σ : ℝ) (d c : Cell)                                    [the parameter, a loop, an injection]
    (h : normForm σ (one - d) ≠ 0)                          [the loop closes]
    mul σ d (restPoint σ d c) + c = restPoint σ d c         [one more trip changes nothing]

    So: `loop_solvable_iff_off_cone` supplies the resolvent and this is what it resolves. The
    hypothesis is the same one that theorem names, so the fixed point exists on exactly the
    points where the loop equation has a solution. -/
theorem restPoint_fixed (σ : ℝ) (d c : Cell) (h : normForm σ (one - d) ≠ 0) :
    mul σ d (restPoint σ d c) + c = restPoint σ d c := by
  have hu := mul_recip_cancel σ (one - d) h
  have hkey : mul σ (one - d) (restPoint σ d c) = c := by
    calc mul σ (one - d) (restPoint σ d c)
        = mul σ (mul σ (one - d) c) (recip σ (one - d)) := by rw [restPoint, mul_assoc]
      _ = mul σ c (mul σ (one - d) (recip σ (one - d))) := by
            rw [mul_comm σ (one - d) c, mul_assoc]
      _ = c := by rw [hu, mul_comm, one_mul]
  have hexp : mul σ (one - d) (restPoint σ d c)
      = restPoint σ d c - mul σ d (restPoint σ d c) := by
    rw [mul_comm, mul_distrib_sub, mul_comm σ (restPoint σ d c) one, one_mul,
      mul_comm σ (restPoint σ d c) d]
  rw [hexp] at hkey
  calc mul σ d (restPoint σ d c) + c
      = mul σ d (restPoint σ d c) + (restPoint σ d c - mul σ d (restPoint σ d c)) := by
        rw [hkey]
    _ = restPoint σ d c := by abel

/-- **The distance to the resting point is one trip's weight, raised to the trip count.**

    (σ : ℝ) (d c s : Cell)                                  [the parameter and the three inputs]
    (h : normForm σ (one - d) ≠ 0)                          [the loop closes]
    (n : ℕ)                                                 [a depth]
    affineIter σ d c s n - restPoint σ d c                  [how far the loop still is]
      = mul σ (cpow σ d n) (s - restPoint σ d c)            [is n trips times the initial gap]

    So: the error of a constant affine loop is a pure geometric sequence in the loop weight,
    the same weight at every trip because the step is constant. Everything below reads off
    this one equation, and `normForm_cpow` is what turns it into a statement about the cone. -/
theorem affineIter_sub_restPoint (σ : ℝ) (d c s : Cell) (h : normForm σ (one - d) ≠ 0)
    (n : ℕ) :
    affineIter σ d c s n - restPoint σ d c = mul σ (cpow σ d n) (s - restPoint σ d c) := by
  induction n with
  | zero => rw [show cpow σ d 0 = one from rfl, one_mul]; rfl
  | succ n ih =>
    have hfix := restPoint_fixed σ d c h
    have hstep : affineIter σ d c s (n + 1) = mul σ d (affineIter σ d c s n) + c := rfl
    have hcp : cpow σ d (n + 1) = mul σ d (cpow σ d n) := rfl
    have hgap : mul σ d (restPoint σ d c) = restPoint σ d c - c := by
      calc mul σ d (restPoint σ d c) = mul σ d (restPoint σ d c) + c - c := by abel
        _ = restPoint σ d c - c := by rw [hfix]
    calc affineIter σ d c s (n + 1) - restPoint σ d c
        = mul σ d (affineIter σ d c s n) + c - restPoint σ d c := by rw [hstep]
      _ = mul σ d (affineIter σ d c s n) - (restPoint σ d c - c) := by abel
      _ = mul σ d (affineIter σ d c s n) - mul σ d (restPoint σ d c) := by rw [hgap]
      _ = mul σ d (affineIter σ d c s n - restPoint σ d c) := (mul_distrib_sub σ d _ _).symm
      _ = mul σ d (mul σ (cpow σ d n) (s - restPoint σ d c)) := by rw [ih]
      _ = mul σ (cpow σ d (n + 1)) (s - restPoint σ d c) := by rw [hcp, mul_assoc]

/-- The first difference of the iterates.

    (σ : ℝ) (d c s : Cell)                                  [the parameter and the three inputs]
    (h : normForm σ (one - d) ≠ 0)                          [the loop closes]
    (n : ℕ)                                                 [a depth]
    affineIter σ d c s (n + 1) - affineIter σ d c s n       [what one more trip moves]
      = mul σ (d - one) (mul σ (cpow σ d n) (s - restPoint σ d c))   [the gap, scaled] -/
theorem affineIter_first_difference (σ : ℝ) (d c s : Cell) (h : normForm σ (one - d) ≠ 0)
    (n : ℕ) :
    affineIter σ d c s (n + 1) - affineIter σ d c s n
      = mul σ (d - one) (mul σ (cpow σ d n) (s - restPoint σ d c)) := by
  have h0 := affineIter_sub_restPoint σ d c s h n
  have h1 := affineIter_sub_restPoint σ d c s h (n + 1)
  rw [show cpow σ d (n + 1) = mul σ d (cpow σ d n) from rfl] at h1
  have hd : mul σ (d - one) (mul σ (cpow σ d n) (s - restPoint σ d c))
      = mul σ (mul σ d (cpow σ d n)) (s - restPoint σ d c)
        - mul σ (cpow σ d n) (s - restPoint σ d c) := by
    simp only [mul, one, Prod.ext_iff, Prod.fst_sub, Prod.snd_sub]
    constructor <;> ring
  rw [hd, ← h1, ← h0]
  abel

/-- **The second difference of the iterates**, which is the quantity the collapse divides by.

    (σ : ℝ) (d c s : Cell)                                  [the parameter and the three inputs]
    (h : normForm σ (one - d) ≠ 0)                          [the loop closes]
    (n : ℕ)                                                 [a depth]
    affineIter σ d c s (n + 2)                              [three consecutive iterates]
      - (affineIter σ d c s (n + 1) + affineIter σ d c s (n + 1))   [combined the Aitken way]
      + affineIter σ d c s n                                [gives the second difference]
      = mul σ (mul σ (d - one) (d - one))                   [which is the loop, one short of]
          (mul σ (cpow σ d n) (s - restPoint σ d c))        [the unit, squared, times the gap]

    So: this names exactly when the division is safe. `normForm_second_difference` below reads
    its size off the three factors, and each factor lying off the cone is one of the three
    hypotheses the real-number statement wrote as `A ≠ 0`, `A ≠ 1` and `S₀ ≠ x*`. -/
theorem affineIter_second_difference (σ : ℝ) (d c s : Cell) (h : normForm σ (one - d) ≠ 0)
    (n : ℕ) :
    affineIter σ d c s (n + 2)
        - (affineIter σ d c s (n + 1) + affineIter σ d c s (n + 1)) + affineIter σ d c s n
      = mul σ (mul σ (d - one) (d - one)) (mul σ (cpow σ d n) (s - restPoint σ d c)) := by
  have h0 := affineIter_sub_restPoint σ d c s h n
  have h1 := affineIter_sub_restPoint σ d c s h (n + 1)
  have h2 := affineIter_sub_restPoint σ d c s h (n + 2)
  rw [show cpow σ d (n + 1) = mul σ d (cpow σ d n) from rfl] at h1
  rw [show cpow σ d (n + 2) = mul σ d (mul σ d (cpow σ d n)) from rfl] at h2
  have hd : mul σ (mul σ (d - one) (d - one)) (mul σ (cpow σ d n) (s - restPoint σ d c))
      = mul σ (mul σ d (mul σ d (cpow σ d n))) (s - restPoint σ d c)
        - (mul σ (mul σ d (cpow σ d n)) (s - restPoint σ d c)
            + mul σ (mul σ d (cpow σ d n)) (s - restPoint σ d c))
        + mul σ (cpow σ d n) (s - restPoint σ d c) := by
    simp only [mul, one, Prod.ext_iff, Prod.fst_sub, Prod.snd_sub, Prod.fst_add, Prod.snd_add]
    constructor <;> ring
  rw [hd, ← h2, ← h1, ← h0]
  abel

/-- **The side condition of the collapse, factored into three sizes.**

    (σ : ℝ) (d c s : Cell)                                  [the parameter and the three inputs]
    (h : normForm σ (one - d) ≠ 0)                          [the loop closes]
    (n : ℕ)                                                 [a depth]
    normForm σ (affineIter σ d c s (n + 2)                  [the size of the second difference]
      - (affineIter σ d c s (n + 1) + affineIter σ d c s (n + 1))
      + affineIter σ d c s n)
      = normForm σ (one - d) ^ 2 * normForm σ d ^ n         [is the loop's own three sizes]
        * normForm σ (s - restPoint σ d c)                  [multiplied together]

    So: **this is the port's whole content.** Over the reals Aitken's formula needs `A ≠ 0`,
    `A ≠ 1` and a seed away from the fixed point, three conditions of the same kind because
    over a field having non-zero size and being non-zero are one condition. Here they separate:
    each of the three is `normForm σ · ≠ 0`, a point lying off the null cone, and
    `normForm_mul` says the product of the three sizes is the size of the thing divided by. So
    the collapse is available exactly off the cone, in all three of its factors at once. -/
theorem normForm_second_difference (σ : ℝ) (d c s : Cell) (h : normForm σ (one - d) ≠ 0)
    (n : ℕ) :
    normForm σ (affineIter σ d c s (n + 2)
        - (affineIter σ d c s (n + 1) + affineIter σ d c s (n + 1)) + affineIter σ d c s n)
      = normForm σ (one - d) ^ 2 * normForm σ d ^ n * normForm σ (s - restPoint σ d c) := by
  have hsym : normForm σ (d - one) = normForm σ (one - d) := by
    simp only [normForm, AnyRing.ringNormForm, one, AnyRing.ringOne, Prod.fst_sub,
      Prod.snd_sub]
    ring
  rw [affineIter_second_difference σ d c s h n, normForm_mul, normForm_mul, normForm_mul,
    normForm_cpow, hsym]
  ring

/-- **THE HEADLINE: three consecutive iterates give the resting point exactly.**

@cite aitken-1926

    (σ : ℝ) (d c s : Cell) (n : ℕ)                          [the parameter, three inputs, a depth]
    (hclose : normForm σ (one - d) ≠ 0)                     [the loop closes]
    (hloop : normForm σ d ≠ 0)                              [the loop weight lies off the cone]
    (hseed : normForm σ (s - restPoint σ d c) ≠ 0)          [so does the initial gap]
    aitken σ (affineIter σ d c s n) (affineIter σ d c s (n + 1))   [the extrapolation of any]
      (affineIter σ d c s (n + 2)) = restPoint σ d c        [three consecutive iterates rests]

    So: the loop past its third value is optional. This is Aitken 1926 carried onto
    `ℝ[x]/(x² − σ)`, and the three hypotheses are the three factors
    `normForm_second_difference` names, each of them one point lying off the null cone. It is
    an identity at every depth `n`, not a limit, so no convergence and no bound on the trip
    count is used. Below zero the three hypotheses collapse back to the predecessor's
    `A ≠ 0`, `A ≠ 1`, `S₀ ≠ x*`, by `normForm_eq_zero_iff_of_neg`; at zero and above they are
    strictly stronger, and `collapse_always_exact_iff_disc_neg` says the gap is real. -/
theorem aitken_exact (σ : ℝ) (d c s : Cell) (n : ℕ) (hclose : normForm σ (one - d) ≠ 0)
    (hloop : normForm σ d ≠ 0) (hseed : normForm σ (s - restPoint σ d c) ≠ 0) :
    aitken σ (affineIter σ d c s n) (affineIter σ d c s (n + 1)) (affineIter σ d c s (n + 2))
      = restPoint σ d c := by
  have h0 := affineIter_sub_restPoint σ d c s hclose n
  have hsym : normForm σ (d - one) = normForm σ (one - d) := by
    simp only [normForm, AnyRing.ringNormForm, one, AnyRing.ringOne, Prod.fst_sub,
      Prod.snd_sub]
    ring
  have hD : normForm σ (mul σ (mul σ (d - one) (d - one))
      (mul σ (cpow σ d n) (s - restPoint σ d c))) ≠ 0 := by
    rw [normForm_mul, normForm_mul, normForm_mul, normForm_cpow, hsym]
    exact mul_ne_zero (mul_ne_zero hclose hclose) (mul_ne_zero (pow_ne_zero n hloop) hseed)
  have hinv := mul_recip_cancel σ _ hD
  have hnum : mul σ (mul σ (d - one) (mul σ (cpow σ d n) (s - restPoint σ d c)))
        (mul σ (d - one) (mul σ (cpow σ d n) (s - restPoint σ d c)))
      = mul σ (mul σ (mul σ (d - one) (d - one))
          (mul σ (cpow σ d n) (s - restPoint σ d c)))
        (mul σ (cpow σ d n) (s - restPoint σ d c)) := by
    simp only [mul, one, Prod.ext_iff, Prod.fst_sub, Prod.snd_sub]
    constructor <;> ring
  rw [aitken, affineIter_first_difference σ d c s hclose n,
    affineIter_second_difference σ d c s hclose n, hnum]
  rw [mul_assoc, mul_comm σ (mul σ (cpow σ d n) (s - restPoint σ d c)) (recip σ _),
    ← mul_assoc, hinv, one_mul, ← h0]
  abel

/-- **The collapse is available for every closing loop exactly on the elliptic face.**

    (σ : ℝ)                                                 [the parameter]
    ∀ d c s : Cell, ∀ n : ℕ                                 [every loop, injection, seed, depth]
    normForm σ (one - d) ≠ 0 → d ≠ 0 →                      [the loop closes and is not nothing]
    s ≠ restPoint σ d c →                                   [the seed is not already at rest]
    aitken σ (affineIter σ d c s n)                         [then the extrapolation of three]
      (affineIter σ d c s (n + 1))                          [consecutive iterates is]
      (affineIter σ d c s (n + 2)) = restPoint σ d c        [the resting point]
    ↔                                                       [exactly when]
    σ < 0                                                   [σ is below zero]

    So: below zero the only point of size zero is the origin, so `d ≠ 0` and a seed away from
    rest are already the off-cone conditions `aitken_exact` needs, and the collapse holds for
    every loop that closes. At zero and above they are not: there is a loop that closes, is not
    the origin, and still has size zero, and against it the extrapolation returns the current
    iterate rather than the resting point. This is `loop_always_solvable_iff_disc_neg` one level
    up: that theorem says which faces let a loop have an answer, this one says which faces let
    three of its values compute that answer. -/
theorem collapse_always_exact_iff_disc_neg (σ : ℝ) :
    (∀ (d c s : Cell) (n : ℕ), normForm σ (one - d) ≠ 0 → d ≠ 0 → s ≠ restPoint σ d c →
      aitken σ (affineIter σ d c s n) (affineIter σ d c s (n + 1))
        (affineIter σ d c s (n + 2)) = restPoint σ d c) ↔ σ < 0 := by
  constructor
  · intro hall
    by_contra hlt
    obtain ⟨ν, hν0, hνn⟩ :=
      (exists_normForm_zero_iff_disc_nonneg σ).mpr (not_lt.mp hlt)
    have hneg : ∀ z : Cell, normForm σ (-z) = normForm σ z := by
      intro z
      simp only [normForm, AnyRing.ringNormForm, Prod.fst_neg, Prod.snd_neg]
      ring
    have hshift : ∀ z : Cell, normForm σ (one - z) = 1 - 2 * z.1 + normForm σ z := by
      intro z
      simp only [normForm, AnyRing.ringNormForm, one, AnyRing.ringOne, Prod.fst_sub,
      Prod.snd_sub]
      ring
    obtain ⟨d, hd0, hdn, hdc⟩ :
        ∃ d : Cell, d ≠ 0 ∧ normForm σ d = 0 ∧ normForm σ (one - d) ≠ 0 := by
      by_cases hb : normForm σ (one - ν) = 0
      · have h1 : 1 - 2 * ν.1 = 0 := by
          have hx := hshift ν
          rw [hb, hνn] at hx
          linarith
        refine ⟨-ν, neg_ne_zero.mpr hν0, (hneg ν).trans hνn, ?_⟩
        rw [hshift (-ν), (hneg ν).trans hνn]
        simp only [Prod.fst_neg]
        intro hcon
        linarith
      · exact ⟨ν, hν0, hνn, hb⟩
    have hrest : restPoint σ d 0 = 0 := by rw [restPoint, mul_comm, mul_origin]
    have hone0 : (one : Cell) ≠ 0 := by
      intro hcon
      have hx := congrArg Prod.fst hcon
      norm_num at hx
    have hiter1 : affineIter σ d 0 one 1 = d := by
      show mul σ d one + 0 = d
      rw [add_zero, mul_comm, one_mul]
    have hdiv : normForm σ (affineIter σ d 0 one (1 + 2)
        - (affineIter σ d 0 one (1 + 1) + affineIter σ d 0 one (1 + 1))
        + affineIter σ d 0 one 1) = 0 := by
      rw [normForm_second_difference σ d 0 one hdc 1, hdn]
      ring
    have hres := hall d 0 one 1 hdc hd0 (by rw [hrest]; exact hone0)
    rw [aitken, recip_on_cone σ _ hdiv, mul_origin, sub_zero, hiter1, hrest] at hres
    exact hd0 hres
  · intro hσ d c s n hclose hd0 hs
    refine aitken_exact σ d c s n hclose ?_ ?_
    · exact fun hn => hd0 ((normForm_eq_zero_iff_of_neg hσ d).mp hn)
    · exact fun hn => sub_ne_zero_of_ne hs ((normForm_eq_zero_iff_of_neg hσ _).mp hn)

/-- **The three faces, on whether two steps are enough.**

    (σ : ℝ)                                                 [the parameter]
    σ < 0 → aitken σ of three affineIter values             [the elliptic face: the]
      = restPoint σ d c                                     [collapse holds in full]
    σ = 0 → ∀ d : Cell, normForm σ d = 0 → cpow σ d 2 = 0   [at zero, a blocking loop dies in
                                                             two trips]
    0 < σ → ∃ d : Cell, normForm σ d = 0                    [above zero, a blocking loop that]
      ∧ normForm σ (one - d) ≠ 0                            [still closes]
      ∧ ∀ n : ℕ, cpow σ d n ≠ 0                             [and never dies out]

    So: two steps suffice on the elliptic face and on no other, and the parabolic face loses
    less than the split one. A loop weight of size zero is one of the two ways the collapse can
    be blocked, and at `σ = 0` every such weight squares to nothing, so `execution_formula`
    answers in two trips instead: the partial sum stops rather than being extrapolated. Above
    zero that fallback is gone. The blocking weight exhibited there closes, so its resting point
    exists, and it never dies out, so the series has no last term, and its size is zero, so the
    extrapolation cannot divide. The resting point is then reachable only from the loop weight
    itself, which is the datum the collapse was to make unnecessary. The other way to be blocked
    is a seed of size zero away from rest, which `normForm_second_difference` shows enters the
    same product, and no face below zero has one. -/
theorem depth_two_trichotomy (σ : ℝ) :
    (σ < 0 → ∀ (d c s : Cell) (n : ℕ), normForm σ (one - d) ≠ 0 → d ≠ 0 →
        s ≠ restPoint σ d c →
        aitken σ (affineIter σ d c s n) (affineIter σ d c s (n + 1))
          (affineIter σ d c s (n + 2)) = restPoint σ d c)
      ∧ (σ = 0 → ∀ d : Cell, normForm σ d = 0 → cpow σ d 2 = 0)
      ∧ (0 < σ → ∃ d : Cell, normForm σ d = 0 ∧ normForm σ (one - d) ≠ 0
          ∧ ∀ n : ℕ, cpow σ d n ≠ 0) := by
  refine ⟨fun hσ => (collapse_always_exact_iff_disc_neg σ).mpr hσ, ?_, ?_⟩
  · rintro rfl d hd
    have h1 : d.1 = 0 := by
      have h2 : d.1 ^ 2 = 0 := by
        simp only [normForm] at hd
        linarith
      exact pow_eq_zero_iff (by norm_num : 2 ≠ 0) |>.mp h2
    simp only [cpow, mul, one, Prod.ext_iff]
    constructor <;> simp [h1]
  · intro hσ
    have hsq : Real.sqrt σ ^ 2 = σ := Real.sq_sqrt hσ.le
    have hval : σ * (1 / Real.sqrt σ) ^ 2 = 1 := by
      rw [div_pow, one_pow, hsq, mul_one_div, div_self (ne_of_gt hσ)]
    have hsne : Real.sqrt σ ≠ 0 := by
      intro hcon
      rw [hcon] at hval
      norm_num at hval
    refine ⟨((1 : ℝ), 1 / Real.sqrt σ), ?_, ?_, ?_⟩
    · show (1 : ℝ) ^ 2 - σ * (1 / Real.sqrt σ) ^ 2 = 0
      linear_combination -hval
    · have hpt : (one : Cell) - ((1 : ℝ), 1 / Real.sqrt σ)
          = ((0 : ℝ), -(1 / Real.sqrt σ)) := by simp [one]
      rw [hpt]
      show (0 : ℝ) ^ 2 - σ * (-(1 / Real.sqrt σ)) ^ 2 ≠ 0
      intro hcon
      have hz : σ * (1 / Real.sqrt σ) ^ 2 = 0 := by linear_combination -hcon
      rw [hval] at hz
      norm_num at hz
    · intro n hcon
      have hup : upper σ ((1 : ℝ), 1 / Real.sqrt σ) = 2 := by
        simp only [upper]
        rw [mul_one_div, div_self hsne]
        norm_num
      have hc := upper_cpow hσ.le ((1 : ℝ), 1 / Real.sqrt σ) n
      rw [hcon, (halves_zero σ).1, hup] at hc
      exact absurd hc.symm (pow_ne_zero n two_ne_zero)

/-- The partial sums of the execution formula, on this carrier.

    (σ : ℝ) (d : Cell) (n : ℕ)                              [the parameter, a loop, a depth]
    mul σ (one - d) (affineIter σ d one 0 n)                [the loop, times the n-th iterate]
      = one - cpow σ d n                                    [telescopes to the unit less n trips]

    So: `loopSum_telescopes` on `ℝ[x]/(x² − σ)`, which identifies `affineIter σ d one 0` as the
    execution formula's own partial sums. The recurrence they satisfy is the affine one with
    the loop weight for its multiplier and the unit for its injection, so the collapse below is
    about Girard's sum rather than about a sequence that resembles it. -/
theorem affineIter_telescopes (σ : ℝ) (d : Cell) (n : ℕ) :
    mul σ (one - d) (affineIter σ d one 0 n) = one - cpow σ d n := by
  induction n with
  | zero =>
    rw [show affineIter σ d one 0 0 = 0 from rfl, show cpow σ d 0 = one from rfl,
      mul_origin, sub_self]
  | succ n ih =>
    have hstep : affineIter σ d one 0 (n + 1)
        = mul σ d (affineIter σ d one 0 n) + one := rfl
    have hswap : mul σ (one - d) (mul σ d (affineIter σ d one 0 n))
        = mul σ d (mul σ (one - d) (affineIter σ d one 0 n)) := by
      rw [← mul_assoc, ← mul_assoc, mul_comm σ (one - d) d]
    rw [hstep, show cpow σ d (n + 1) = mul σ d (cpow σ d n) from rfl, mul_distrib_add, hswap,
      ih, mul_distrib_sub]
    simp only [mul, one, Prod.ext_iff, Prod.fst_sub, Prod.snd_sub, Prod.fst_add, Prod.snd_add]
    constructor <;> ring

/-- **The resolvent, from three partial sums of the execution formula.**

@cite aitken-1926
@cite girard-1989

    (σ : ℝ) (d : Cell) (n : ℕ)                              [the parameter, a loop, a depth]
    (hclose : normForm σ (one - d) ≠ 0)                     [the loop closes]
    (hloop : normForm σ d ≠ 0)                              [the loop weight lies off the cone]
    aitken σ (affineIter σ d one 0 n)                       [the extrapolation of three]
      (affineIter σ d one 0 (n + 1))                        [consecutive partial sums]
      (affineIter σ d one 0 (n + 2)) = recip σ (one - d)    [is the resolvent itself]

    So: this is the payoff `execution_formula` was missing. That theorem needs the loop to be
    nilpotent, which `nilpotent_iff_disc_zero` restricts to `σ = 0`; this one needs no
    nilpotency and no depth bound, and returns the inverse `loop_solvable_iff_off_cone` asserts
    exists. The third hypothesis `aitken_exact` carries is discharged here rather than assumed:
    the seed is the origin, so the initial gap is the resolvent, whose size is the reciprocal
    of the loop's own by `normForm_recip` and so is never zero. Two steps past the first
    therefore resolve the series, and `feedback a b c` of the result is the value of the whole
    block. -/
theorem resolvent_from_three_partial_sums (σ : ℝ) (d : Cell) (n : ℕ)
    (hclose : normForm σ (one - d) ≠ 0) (hloop : normForm σ d ≠ 0) :
    aitken σ (affineIter σ d one 0 n) (affineIter σ d one 0 (n + 1))
      (affineIter σ d one 0 (n + 2)) = recip σ (one - d) := by
  have hR : restPoint σ d one = recip σ (one - d) := by rw [restPoint, one_mul]
  have hseed : normForm σ ((0 : Cell) - restPoint σ d one) ≠ 0 := by
    rw [hR, show (0 : Cell) - recip σ (one - d) = -recip σ (one - d) by abel]
    have hn : normForm σ (-recip σ (one - d)) = normForm σ (recip σ (one - d)) := by
      simp only [normForm, AnyRing.ringNormForm, Prod.fst_neg, Prod.snd_neg]
      ring
    rw [hn, normForm_recip σ (one - d) hclose]
    exact inv_ne_zero hclose
  rw [← hR]
  exact aitken_exact σ d one 0 n hclose hloop hseed

-- ═══════════════════════════════════════════════════════════════════
-- THE REPLICATE FOLD, AND THE COLLAPSE READ ON PREFIX STATES
-- ═══════════════════════════════════════════════════════════════════

/-- The iterate shifted by one is the iterate from the stepped seed.

    (sigma : ℝ)                       [the parameter]
    (d c s : Cell)                    [the loop weight, the injection, the seed]
    (m : ℕ)                           [the number of trips after the shift]
    affineIter sigma d c s (m + 1)    [the loop after one trip and then m more]
    = affineIter sigma d c (AnyRing.ringAffineApply sigma ⟨d, c⟩ s) m
                                      [the same trips, from the seed the one step produced]
    `affineIter`                      [the affine loop run that many times: multiply by the]
    `AnyRing.ringAffineApply`         [apply one affine step: multiply by the coefficient]
    `Cell`                            [a point: two reals side by side, a + b·x]
    `ℝ`                               [the real numbers]
    `ℕ`                               [the natural numbers]

    So: both sides are the step applied `m + 1` times to the seed, with the recurrence unfolded
    once on the left. -/
private theorem affineIter_succ_apply (sigma : ℝ) (d c s : Cell) (m : ℕ) :
    affineIter sigma d c s (m + 1) =
      affineIter sigma d c (AnyRing.ringAffineApply sigma ⟨d, c⟩ s) m := by
  induction m generalizing s with
  | zero => rfl
  | succ m ih =>
    show AnyRing.ringAffineApply sigma ⟨d, c⟩ (affineIter sigma d c s (m + 1)) =
      AnyRing.ringAffineApply sigma ⟨d, c⟩
        (affineIter sigma d c (AnyRing.ringAffineApply sigma ⟨d, c⟩ s) m)
    rw [ih s]

/-- **The iterate is the replicate fold: running the loop n times applies the composed step.**

    (sigma : ℝ)                     [the parameter]
    (d c s : Cell)                  [the loop weight, the injection, the seed]
    (n : ℕ)                         [the number of trips round the loop]
    affineIter sigma d c s n        [the affine loop run that many times]
    =                               [is equal to]
    AnyRing.ringAffineApply sigma
      (AnyRing.ringAffineFold sigma (List.replicate n ⟨d, c⟩)) s
                                    [the replicate fold of the step applied once]
    `affineIter`                    [the affine loop run that many times: multiply by the]
    `AnyRing.ringAffineApply`       [apply one affine step: multiply by the coefficient]
    `AnyRing.ringAffineFold`        [the left fold of a list of steps into one step]
    `List`                          [a finite sequence of things, in order]
    `Cell`                          [a point: two reals side by side, a + b·x]
    `ℝ`                             [the real numbers]
    `ℕ`                             [the natural numbers]

    So: `AnyRing.ringAffineFold_apply` turns the fold into the left fold of applies, and the
    left fold of a constant replicate is the iterate by induction. -/
theorem affineIter_eq_ringAffineFold (sigma : ℝ) (d c s : Cell) (n : ℕ) :
    affineIter sigma d c s n =
      AnyRing.ringAffineApply sigma
        (AnyRing.ringAffineFold sigma (List.replicate n ⟨d, c⟩)) s := by
  rw [AnyRing.ringAffineFold_apply]
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    rw [List.replicate_succ, List.foldl_cons,
      ← ih (AnyRing.ringAffineApply sigma ⟨d, c⟩ s)]
    exact affineIter_succ_apply sigma d c s n

/-- **Aitken on three prefix folds is the resting point, at every depth.**

    (sigma : ℝ) (d c s : Cell) (n : ℕ)      [the parameter, loop weight, injection, seed, depth]
    (hclose : normForm sigma (one - d) ≠ 0) [the loop closes]
    (hloop : normForm sigma d ≠ 0)          [the loop weight lies off the cone]
    (hseed : normForm sigma (s - restPoint sigma d c) ≠ 0) [so does the initial gap]
    aitken sigma of the three prefix folds  [the extrapolation of three consecutive states]
    = restPoint sigma d c                   [is the resting point]
    `aitken`                                [the first of three points, less their first]
    `normForm`                              [the norm form: a point times its conjugate]
    `one`                                   [the point (1, 0), which changes nothing]
    `restPoint`                             [where the affine loop comes to rest]
    `AnyRing.ringAffineApply`               [apply one affine step: multiply by the coefficient]
    `AnyRing.ringAffineFold`                [the left fold of a list of steps into one step]
    `List`                                  [a finite sequence of things, in order]
    `Cell`                                  [a point: two reals side by side, a + b·x]
    `ℝ`                                     [the real numbers]
    `ℕ`                                     [the natural numbers]
    `≠`                                     [is not equal to]

    So: `affineIter_eq_ringAffineFold` rewrites each prefix fold to the iterate it names, and
    `aitken_exact` gives the resting point under its three existing hypotheses. The fixed-point
    proof is not restated. -/
theorem aitken_exact_of_prefix_fold (sigma : ℝ) (d c s : Cell) (n : ℕ)
    (hclose : normForm sigma (one - d) ≠ 0) (hloop : normForm sigma d ≠ 0)
    (hseed : normForm sigma (s - restPoint sigma d c) ≠ 0) :
    aitken sigma
      (AnyRing.ringAffineApply sigma
        (AnyRing.ringAffineFold sigma (List.replicate n ⟨d, c⟩)) s)
      (AnyRing.ringAffineApply sigma
        (AnyRing.ringAffineFold sigma (List.replicate (n + 1) ⟨d, c⟩)) s)
      (AnyRing.ringAffineApply sigma
        (AnyRing.ringAffineFold sigma (List.replicate (n + 2) ⟨d, c⟩)) s)
      = restPoint sigma d c := by
  rw [← affineIter_eq_ringAffineFold sigma d c s n,
    ← affineIter_eq_ringAffineFold sigma d c s (n + 1),
    ← affineIter_eq_ringAffineFold sigma d c s (n + 2)]
  exact aitken_exact sigma d c s n hclose hloop hseed

/-- **At the parabolic setting a loop weight of size zero squares away in two trips.**

    (d c s : Cell)                   [the loop weight, the injection, the seed]
    (hd : normForm 0 d = 0)          [the loop weight has size zero at the parabolic setting]
    AnyRing.ringAffineApply 0
      (AnyRing.ringAffineFold 0 (List.replicate 2 (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s
                                     [run the two-step replicate fold from the seed]
    = mul 0 d c + c                  [is the weight times the injection, plus the injection]
    `normForm`                       [the norm form: a point times its conjugate]
    `mul`                            [multiply two points together, the σ way]
    `List`                           [a finite sequence of things, in order]
    `Cell`                           [a point: two reals side by side, a + b·x]
    `ℝ`                              [the real numbers]

    So: `normForm 0 d = 0` forces the first slot of `d` to zero, so `d · d` is the origin and
    the two trips collapse to `d · c + c`. The seed and the injection are arbitrary. -/
theorem parabolic_square_zero_prefix_fold (d c s : Cell) (hd : normForm 0 d = 0) :
    AnyRing.ringAffineApply 0
      (AnyRing.ringAffineFold 0 (List.replicate 2 (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s =
    mul 0 d c + c := by
  rw [AnyRing.ringAffineFold_apply]
  simp only [List.replicate_succ, List.replicate_zero, List.foldl_cons, List.foldl_nil]
  have hd1 : d.1 = 0 := by
    have h2 : d.1 ^ 2 = 0 := by
      simp only [normForm, AnyRing.ringNormForm] at hd
      linarith
    exact pow_eq_zero_iff (by norm_num : 2 ≠ 0) |>.mp h2
  have hdd : AnyRing.ringMul (0 : ℝ) d d = 0 := by
    ext <;> simp [hd1]
  simp only [AnyRing.ringAffineApply]
  rw [AnyRing.ringMul_distrib_add, ← AnyRing.ringMul_assoc, hdd,
    AnyRing.ringMul_comm (0 : ℝ) 0 s, AnyRing.ringMul_zero, zero_add]

/-- **The collapse is available for every closing loop on the prefix-fold states exactly below
    zero.**

    (σ : ℝ)                                                 [the parameter]
    ∀ d c s : Cell, ∀ n : ℕ                                 [every loop, injection, seed, depth]
    normForm σ (one - d) ≠ 0 → d ≠ 0 →                      [the loop closes and is not nothing]
    s ≠ restPoint σ d c →                                   [the seed is not already at rest]
    aitken σ of the three prefix folds                      [then the extrapolation of three]
      = restPoint σ d c                                     [consecutive states is the rest point]
    ↔                                                       [exactly when]
    σ < 0                                                   [σ is below zero]
    `aitken`                                                [the first of three points, less their]
    `normForm`                                              [the norm form: a point times its]
    `one`                                                   [the point (1, 0), which changes]
    `restPoint`                                             [where the affine loop comes to rest]
    `List`                                                  [a finite sequence of things, in]
    `Cell`                                                  [a point: two reals side by side,]
    `ℝ`                                                     [the real numbers]
    `ℕ`                                                     [the counting numbers 0, 1, 2, ...]

    So: `collapse_always_exact_iff_disc_neg` with every state rewritten by
    `affineIter_eq_ringAffineFold`, so the same iff reads on the replicate fold the scan
    computes. -/
theorem collapse_always_exact_iff_disc_neg_prefix_fold (σ : ℝ) :
    (∀ (d c s : Cell) (n : ℕ), normForm σ (one - d) ≠ 0 → d ≠ 0 →
      s ≠ restPoint σ d c →
      aitken σ
        (AnyRing.ringAffineApply σ
          (AnyRing.ringAffineFold σ (List.replicate n (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s)
        (AnyRing.ringAffineApply σ
          (AnyRing.ringAffineFold σ
            (List.replicate (n + 1) (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s)
        (AnyRing.ringAffineApply σ
          (AnyRing.ringAffineFold σ
            (List.replicate (n + 2) (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s)
        = restPoint σ d c) ↔ σ < 0 := by
  rw [← collapse_always_exact_iff_disc_neg σ]
  constructor
  · intro h d c s n hc hd hs
    rw [affineIter_eq_ringAffineFold σ d c s n,
      affineIter_eq_ringAffineFold σ d c s (n + 1),
      affineIter_eq_ringAffineFold σ d c s (n + 2)]
    exact h d c s n hc hd hs
  · intro h d c s n hc hd hs
    rw [← affineIter_eq_ringAffineFold σ d c s n,
      ← affineIter_eq_ringAffineFold σ d c s (n + 1),
      ← affineIter_eq_ringAffineFold σ d c s (n + 2)]
    exact h d c s n hc hd hs

/-- **The positive face has a blocking loop whose prefix folds defeat the collapse.**

    ∃ sigma : ℝ                                     [there is a parameter]
    0 < sigma                                       [it is on the positive face]
    ∧ ∃ d c s : Cell, ∃ n : ℕ                       [and there is a loop, injection, seed, depth]
    normForm sigma (one - d) ≠ 0                    [the loop closes]
    ∧ d ≠ 0                                         [the loop weight is not the origin]
    ∧ s ≠ restPoint sigma d c                       [the seed is not already at rest]
    ∧ aitken sigma of the three prefix folds        [and the extrapolation of three]
      ≠ restPoint sigma d c                         [consecutive states misses the rest point]
    `normForm`                                      [the norm form: a point times its conjugate]
    `one`                                           [the point (1, 0), which changes nothing]
    `restPoint`                                     [where the affine loop comes to rest]
    `aitken`                                        [the first of three points, less their first]
    `List`                                          [a finite sequence of things, in order]
    `Cell`                                          [a point: two reals side by side, a + b·x]
    `ℝ`                                             [the real numbers]
    `ℕ`                                             [the counting numbers 0, 1, 2, and so on]
    `∃`                                             [there is at least one]
    `∧`                                             [both are true]
    `<`                                             [is smaller than]
    `≠`                                             [is not equal to]

    So: `collapse_always_exact_iff_disc_neg_prefix_fold` fails at a positive setting, and the
    witness is `sigma = 1`, `d = (1, 1)`, `c = 0`, `s = one`, `n = 1`, read by `norm_num`. The
    three prefix folds are `(1, 1)`, `(2, 2)` and `(4, 4)`, their second difference has size zero,
    so the extrapolation returns the first state rather than the resting point. -/
theorem positive_face_prefix_fold_failure :
    ∃ sigma : ℝ, 0 < sigma ∧
    ∃ (d c s : Cell) (n : ℕ),
      normForm sigma (one - d) ≠ 0 ∧ d ≠ 0 ∧ s ≠ restPoint sigma d c ∧
      aitken sigma
        (AnyRing.ringAffineApply sigma
          (AnyRing.ringAffineFold sigma
            (List.replicate n (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s)
        (AnyRing.ringAffineApply sigma
          (AnyRing.ringAffineFold sigma
            (List.replicate (n + 1) (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s)
        (AnyRing.ringAffineApply sigma
          (AnyRing.ringAffineFold sigma
            (List.replicate (n + 2) (⟨d, c⟩ : AnyRing.RingAffineStep ℝ))) s)
        ≠ restPoint sigma d c := by
  refine ⟨1, by norm_num, ((1 : ℝ), (1 : ℝ)), 0, one, 1, ?_, ?_, ?_, ?_⟩
  · norm_num [normForm, one, AnyRing.ringNormForm, AnyRing.ringOne]
  · norm_num
  · norm_num [one, AnyRing.ringOne, restPoint, AnyRing.ringMul, AnyRing.ringMul_zero,
      AnyRing.ringMul_comm]
  · norm_num [aitken, restPoint, normForm, one, recip, AnyRing.ringNormForm,
      AnyRing.ringOne, AnyRing.ringMul, AnyRing.ringAffineApply, AnyRing.ringAffineFold,
      AnyRing.ringAffineCompose, AnyRing.ringAffineId, List.replicate, List.foldl,
      List.foldl_cons, List.foldl_nil]

/-- **The real split-face witness: three sizes, the resting point, and the collapse.**

    (n : ℕ)                        [the depth at which the collapse is read]
    (1 : ℝ)                        [the parameter at the split face, one]
    (3, 1)                         [the loop weight: first slot 3, second slot 1]
    (1, 0)                         [the injection: first slot 1, second slot 0]
    (0, 0)                         [the seed: the origin]
    ∧                              [all of the following hold]
    `normForm`                     [the norm form: a point times its conjugate]
    `restPoint`                    [where the affine loop comes to rest]
    `aitken`                       [the first of three points, less their first difference]
    `affineIter`                   [the affine loop run that many times: multiply by the]
    `AnyRing.ringUpper`            [the first-coordinate read over a ring]
    `AnyRing.ringLower`            [the second-coordinate read over a ring]
    `one`                          [the point (1, 0), which changes nothing]
    `ℝ`                            [the real numbers]
    `ℕ`                            [the natural numbers]
    `≠`                            [is not equal to]

    So: the loop weight `(3, 1)` and injection `(1, 0)` at `sigma = 1` give the three sizes `3`,
    `8` and `1/3`, the resting point `(-2/3, 1/3)`, a non-zero second slot and two unequal
    channels. The collapse is `aitken_exact` with those three hypotheses discharged, so it holds
    at every `n` rather than being assumed. -/
theorem aitken_hyperbolic_witness (n : ℕ) :
    normForm (1 : ℝ) (one - ((3 : ℝ), (1 : ℝ))) = 3 ∧
    normForm (1 : ℝ) ((3 : ℝ), (1 : ℝ)) = 8 ∧
    normForm (1 : ℝ)
        (((0 : ℝ), (0 : ℝ))
          - restPoint (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ))) = 1 / 3 ∧
    restPoint (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ)) = ((-2 / 3 : ℝ), (1 / 3 : ℝ)) ∧
    ((3 : ℝ), (1 : ℝ)).2 ≠ 0 ∧
    AnyRing.ringUpper (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ≠
      AnyRing.ringLower (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ∧
    aitken (1 : ℝ)
      (affineIter (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ)) ((0 : ℝ), (0 : ℝ)) n)
      (affineIter (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ)) ((0 : ℝ), (0 : ℝ)) (n + 1))
      (affineIter (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ)) ((0 : ℝ), (0 : ℝ)) (n + 2))
      = restPoint (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ)) := by
  have hrest : restPoint (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ))
      = ((-2 / 3 : ℝ), (1 / 3 : ℝ)) := by
    norm_num [restPoint, recip, normForm, mul, one, AnyRing.ringMul, AnyRing.ringOne,
      AnyRing.ringNormForm]
  refine ⟨?_, ?_, ?_, hrest, ?_, ?_, ?_⟩
  · norm_num [normForm, one, AnyRing.ringNormForm, AnyRing.ringOne]
  · norm_num [normForm, AnyRing.ringNormForm]
  · rw [hrest]
    norm_num [normForm, AnyRing.ringNormForm]
  · norm_num
  · norm_num [AnyRing.ringUpper, AnyRing.ringLower]
  · exact aitken_exact (1 : ℝ) ((3 : ℝ), (1 : ℝ)) ((1 : ℝ), (0 : ℝ)) ((0 : ℝ), (0 : ℝ)) n
      (by norm_num [normForm, one, AnyRing.ringNormForm, AnyRing.ringOne])
      (by norm_num [normForm, AnyRing.ringNormForm])
      (by rw [hrest]; norm_num [normForm, AnyRing.ringNormForm])

end Trichotomy
end Apeiron

end
