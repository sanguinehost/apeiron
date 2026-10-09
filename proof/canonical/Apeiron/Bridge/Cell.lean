--//@ file: Apeiron/Bridge/Cell.lean
--//@ does: bridge Lean Apeiron Cell to song decider interval cell
--//@ ring: R0
--//@ onto: asset:primitive:interval_bridge
--//@ cite: Clifford (1873), Kaucher (1980)
--//@ grade: proved

import Apeiron.Trichotomy.NullBasis

/-!
# The bridge to song's cell

@name The Cell Bridge
@grade proved

`../song/src/prover/cone/cell.slang` claims its decider's interval pair *is* an element of this
algebra at `σ = 1`, held in the null basis, and proves that claim in-tree as `law` blocks over
bounded integers. This file states those laws in Lean and derives them from the corpus's
σ-parametric theorems, so the two proofs are the same proof rather than two proofs that agree.

## The correspondence

song holds an interval as a pair `(lo, hi)` and moves to doubled cell coordinates, because the
halving in a change of basis does not stay in the integers:

```
   cellMid2 (lo, hi) = hi + lo        cellRad2 (lo, hi) = hi - lo
```

`ofBounds` is that map into `Cell`. Under it, song's `upperOf` and `lowerOf` are the corpus's
`upper 1` and `lower 1`, and the doubling song carries for integrality is the factor of two its
own laws state.

## What this settles, and what it does not

It settles that song's cell laws follow from theorems here. It does **not** settle that the
compiled Slang program computes what its laws say: that is the refinement gap song records at
`CAVEATS.md`, and it needs a Slang semantics or a certificate replay, not a theorem. `song`'s own
terminal invariant is that the prover proves everything, so nothing here is load-bearing inside
song; this is the spec side agreeing with the decider side, checked.
-/

namespace Apeiron
namespace Bridge

open Trichotomy

/-- An interval, as song holds it, read into the algebra.

    (lo hi : ℤ)                       [the two endpoints, as whole numbers]
    : Cell                            [the point whose doubled coordinates they are]

    So: `cellMid2` and `cellRad2` of `../song/src/prover/cone/cell.slang`, doubled exactly as
    that file doubles them so the change of basis stays in the integers. -/
def ofBounds (lo hi : ℤ) : Cell := (((hi + lo : ℤ) : ℝ), ((hi - lo : ℤ) : ℝ))

/-- song's `upperOf` is the corpus's upper channel, at the split face.

    (lo hi : ℤ)                                  [the two endpoints, as whole numbers]
    upper 1 (ofBounds lo hi) = 2 * (hi : ℝ)      [twice the top endpoint, as a real number]

    So: `law cell_upper_is_hi`, derived rather than restated. -/
theorem upper_ofBounds (lo hi : ℤ) : upper 1 (ofBounds lo hi) = 2 * (hi : ℝ) := by
  simp only [upper, ofBounds, Real.sqrt_one]
  push_cast
  ring

/-- song's `lowerOf` is the corpus's lower channel, at the split face.

    (lo hi : ℤ)                                  [the two endpoints, as whole numbers]
    lower 1 (ofBounds lo hi) = 2 * (lo : ℝ)      [twice the bottom endpoint, as a real number]

    So: `law cell_lower_is_lo`. -/
theorem lower_ofBounds (lo hi : ℤ) : lower 1 (ofBounds lo hi) = 2 * (lo : ℝ) := by
  simp only [lower, ofBounds, Real.sqrt_one]
  push_cast
  ring

/-- Conjugation swaps the endpoints.

    (lo hi : ℤ)                                     [the two endpoints]
    conj (ofBounds lo hi) = ofBounds hi lo          [conjugating exchanges them]

    So: song's `conjLower` and `conjUpper` in one statement. Its four conjugation laws
    (`cell_conj_lower_is_upper`, `cell_conj_upper_is_lower`, and the two scaling laws that select
    a channel by the sign of a coefficient) are this theorem read on each channel. The corpus's
    `conj` is the algebra's own involution, so the endpoint swap a propagator performs on a
    negative coefficient is not a case in the propagator, it is the algebra. -/
theorem conj_ofBounds (lo hi : ℤ) : conj (ofBounds lo hi) = ofBounds hi lo := by
  simp only [conj, ofBounds, Prod.ext_iff]
  constructor <;> · push_cast; ring

/-- Conjugating twice returns the interval.

    (lo hi : ℤ)                                            [the two endpoints]
    conj (conj (ofBounds lo hi)) = ofBounds lo hi          [the swap is an involution]

    So: `law cell_conj_involution`, which here is `conj_conj` at this point rather than a
    separate computation. -/
theorem conj_conj_ofBounds (lo hi : ℤ) :
    conj (conj (ofBounds lo hi)) = ofBounds lo hi := conj_conj _

/-- The size of an interval's cell is four times the product of its endpoints.

    (lo hi : ℤ)                                             [the two endpoints]
    normForm 1 (ofBounds lo hi) = 4 * (lo : ℝ) * hi         [four times the endpoints multiplied]

    So: `normForm_eq_upper_mul_lower` at this point, with the two factors of two the doubling
    carries. song's `normRec` builds the same product by repeated addition so its bounds come out
    by induction. -/
theorem normForm_ofBounds (lo hi : ℤ) :
    normForm 1 (ofBounds lo hi) = 4 * (lo : ℝ) * (hi : ℝ) := by
  rw [normForm_eq_upper_mul_lower (by norm_num), upper_ofBounds, lower_ofBounds]
  ring

/-- An interval is a crossing exactly when its endpoints have passed each other.

    (lo hi : ℤ)                                    [the two endpoints]
    Crossing 1 (ofBounds lo hi) ↔ (hi : ℝ) < lo    [the top has fallen below the bottom]

    So: the condition a propagator reports as unsatisfiable, read on the algebra. -/
theorem crossing_ofBounds (lo hi : ℤ) :
    Crossing 1 (ofBounds lo hi) ↔ (hi : ℝ) < (lo : ℝ) := by
  simp only [Crossing, upper_ofBounds, lower_ofBounds]
  constructor <;> intro h <;> linarith

/-- Crossing and lying on the cone are different conditions, on song's own carrier.

    (lo hi : ℤ)                                          [the two endpoints]
    Crossing 1 (ofBounds lo hi) ↔ (hi : ℝ) < lo          [a crossing is the endpoints passing]
    normForm 1 (ofBounds lo hi) = 0 ↔ lo = 0 ∨ hi = 0    [the cone is an endpoint at zero]
    ∧                                                    [both together]

    So: `../song/src/prover/cone/cell.slang`'s C4, stated over the reals against `Cell σ` rather
    than over integers under a cap. The two loci are visibly different sets: `lo = 2, hi = 1` is a
    crossing off the cone, and `lo = 0, hi = 1` is on the cone and not a crossing. song proves
    both directions by exhibiting those two cases; here they are read off the equivalences. -/
theorem crossing_independent_of_cone_ofBounds (lo hi : ℤ) :
    (Crossing 1 (ofBounds lo hi) ↔ (hi : ℝ) < (lo : ℝ)) ∧
      (normForm 1 (ofBounds lo hi) = 0 ↔ lo = 0 ∨ hi = 0) := by
  refine ⟨crossing_ofBounds lo hi, ?_⟩
  rw [normForm_ofBounds]
  constructor
  · intro h
    rcases mul_eq_zero.mp h with h' | h'
    · rcases mul_eq_zero.mp h' with h'' | h''
      · norm_num at h''
      · exact Or.inl (by exact_mod_cast h'')
    · exact Or.inr (by exact_mod_cast h')
  · rintro (h | h) <;> rw [h] <;> push_cast <;> ring

/-- Propagation keeps exactly the numbers both intervals keep.

    {l₁ h₁ l₂ h₂ : ℤ} {t : ℝ}                                       [two intervals and a number]
    InCell 1 t (meet 1 (ofBounds l₁ h₁) (ofBounds l₂ h₂))           [the meet holds it]
    ↔                                                               [exactly when]
    InCell 1 t (ofBounds l₁ h₁) ∧ InCell 1 t (ofBounds l₂ h₂)       [both intervals hold it]

    So: song's `cell_meet_lower_sound` and `cell_meet_upper_sound` are the right-to-left
    direction, and the left-to-right direction is the tightness its `meetLowerTight` and
    `meetUpperTight` laws state separately. Here it is one equivalence. -/
theorem meet_ofBounds_iff {l₁ h₁ l₂ h₂ : ℤ} {t : ℝ} :
    InCell 1 t (meet 1 (ofBounds l₁ h₁) (ofBounds l₂ h₂))
      ↔ InCell 1 t (ofBounds l₁ h₁) ∧ InCell 1 t (ofBounds l₂ h₂) :=
  inCell_meet_iff (by norm_num) t _ _

/-- Propagation is the interval of the tighter bounds.

    (l₁ h₁ l₂ h₂ : ℤ)                                   [two intervals, by their endpoints]
    meet 1 (ofBounds l₁ h₁) (ofBounds l₂ h₂)            [meeting them in the algebra]
      = ofBounds (max l₁ l₂) (min h₁ h₂)                [is the interval of the tighter bounds]

    So: `../song/src/prover/cone/ncone0.slang`'s propagation step, in one statement. Its
    `prop_newlo_raises` and `prop_newlo_keeps` say the lower endpoint takes the larger of the two,
    its `prop_newhi_lowers` and `prop_newhi_keeps` say the upper takes the smaller, and here both
    are what `upper_meet` and `lower_meet` already say about the two channels. song carries these
    as laws on the running decider because its spec faces cannot be linked; this is the spec side
    of the same fact. -/
theorem meet_ofBounds (l₁ h₁ l₂ h₂ : ℤ) :
    meet 1 (ofBounds l₁ h₁) (ofBounds l₂ h₂) = ofBounds (max l₁ l₂) (min h₁ h₂) := by
  have hpos : (0 : ℝ) < 1 := by norm_num
  have hu : upper 1 (meet 1 (ofBounds l₁ h₁) (ofBounds l₂ h₂)) = 2 * ((min h₁ h₂ : ℤ) : ℝ) := by
    rw [upper_meet hpos, upper_ofBounds, upper_ofBounds]
    push_cast
    rcases le_total (h₁ : ℝ) h₂ with h | h
    · rw [min_eq_left h, min_eq_left (by linarith)]
    · rw [min_eq_right h, min_eq_right (by linarith)]
  have hl : lower 1 (meet 1 (ofBounds l₁ h₁) (ofBounds l₂ h₂)) = 2 * ((max l₁ l₂ : ℤ) : ℝ) := by
    rw [lower_meet hpos, lower_ofBounds, lower_ofBounds]
    push_cast
    rcases le_total (l₁ : ℝ) l₂ with h | h
    · rw [max_eq_right h, max_eq_right (by linarith)]
    · rw [max_eq_left h, max_eq_left (by linarith)]
  have hrt := ofHalves_upper_lower hpos (meet 1 (ofBounds l₁ h₁) (ofBounds l₂ h₂))
  rw [hu, hl] at hrt
  rw [← hrt]
  simp only [ofHalves, ofBounds, Real.sqrt_one, Prod.ext_iff]
  constructor <;> · push_cast; ring

/-- Selecting an endpoint by the sign of a coefficient is the algebra's conjugation.

    (lo hi : ℤ)                                                [an interval]
    upper 1 (conj (ofBounds lo hi)) = lower 1 (ofBounds lo hi)  [conjugating swaps the readings]

    So: `../song/src/prover/cone/ncone0.slang`'s `prop_pick_upper` and `prop_pick_lower`, which
    read the top endpoint for a positive coefficient and the bottom for a negative one. song's
    `cell.slang` calls this C2 and says the swap is the involution the quotient carries rather
    than a case inside the propagator. This is that, discharged. -/
theorem upper_conj_ofBounds (lo hi : ℤ) :
    upper 1 (conj (ofBounds lo hi)) = lower 1 (ofBounds lo hi) := by
  rw [conj_ofBounds, upper_ofBounds, lower_ofBounds]

end Bridge
end Apeiron
