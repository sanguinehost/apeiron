import Apeiron.Trichotomy.Trace
import Mathlib.LinearAlgebra.Trace
import Mathlib.Logic.Equiv.Nat

/-!
# The carrier a feedback calculus needs, and why it is not two-dimensional

@name Wires
@cite girard-1989
@cite abramsky-haghverdi-scott-2002
@cite joyal-street-verity-1996
@cite palmieri-2026
@grade proved

Girard's application is composition with feedback, and `Trichotomy/Trace.lean` builds that
feedback on the algebra. What it does not have is a carrier to run it on. Abramsky, Haghverdi
and Scott's construction needs one object `U` that contains two copies of itself, so that a
pair of arguments can travel down a single wire and be separated again at the other end. This
file shows that a two-dimensional carrier cannot be that object, and builds one that can.

## Root

| from | here |
|---|---|
| `classification`, `dim_two` | `no_two_wire_split_on_cell` |
| `two_channels_iff_disc_pos`, `upper_eq_lower_of_disc_zero` | `wire_reads_two_reals_iff_disc_pos` |
| `mul` at any σ | `leftIn_weighted`, `rightIn_weighted` |

The negative comes first because it is what forces the rest. `no_two_wire_split_on_cell` says
no four linear maps on a two-dimensional space satisfy the two-wire equations, and the proof is
a trace count: each retraction contributes the whole dimension, so the two of them contribute
twice what the identity does. This is why the calculus cannot live on `Cell` no matter which
face is chosen, and it is a different obstruction from the one in `theory/DEAD_ENDS.md`. That
one is about associativity and applies to every carrier. This one is about size and applies to
this carrier at every σ.

`wire_reads_two_reals_iff_disc_pos` is where σ re-enters. A wire carries a point of the
algebra at each position, and that point carries two independent real readings exactly when
σ is above zero. So on the split face one algebra-weighted wire is two real wires, and at or
below zero it is one, which is `two_channels_iff_disc_pos` moved onto the carrier.

## What is built

`ℕ → Cell` with the standard pairing of the counting numbers. `leftIn` and `rightIn` place a
wire in the even and odd positions, `leftOut` and `rightOut` read them back, and the five
equations `leftOut_leftIn` through `in_out_id` are the partial isometry conditions Abramsky,
Haghverdi and Scott ask of the retraction `U ⊕ U ◁ U`. `leftIn_weighted` says the placements
commute with an algebra weight, so the wires carry `mul σ` along them and `+` across them,
which is the shape a feedback sum needs.

## What is built next, and where the construction stops

@cite abramsky-haghverdi-scott-2002

Abramsky, Haghverdi and Scott, *Geometry of Interaction and linear combinatory algebras*,
Mathematical Structures in Computer Science 12 (2002) 625-665, prove that a GoI situation yields
a linear combinatory algebra. A GoI situation is a traced category with a reflexive object and a
traced functor `T` standing for the exponential of linear logic, and it is `T` that supplies the
duplication `S` needs.

`Combinator/Interaction.lean` builds both over this carrier, concretely and with no abstract
comonad anywhere: the reflexive object is the even and odd positions, and `T X = ℕ × X` is
Cantor's pairing, with `uncopy_copy` and `uncopy_copy_of_ne` proving the retraction and the
disjointness of the countably many copies. So the exponential is a construction here rather than
a missing library object.

What that file then names, rather than claims, is the assembly: the combinator equations under
the traced application need the trace as a union over paths of unbounded length, which is an
induction over the loop wire that is not yet written. The equation is stated there.

-/

noncomputable section

namespace Apeiron
namespace Combinator

open Trichotomy

-- ═══════════════════════════════════════════════════════════════════
-- WHY THE ALGEBRA'S OWN CARRIER CANNOT CARRY THE CALCULUS
-- ═══════════════════════════════════════════════════════════════════

/-- **A two-dimensional space holds no two copies of itself.**

@cite palmieri-2026
@cite abramsky-haghverdi-scott-2002

    ¬ ∃ l r pl pr : Cell →ₗ[ℝ] Cell               [there are no four linear maps on points]
    pl * l = LinearMap.id                          [the first pair being a retraction]
    pr * r = LinearMap.id                          [the second pair too]
    l * pl + r * pr = LinearMap.id                 [and the two images filling the space]
    ∧                                              [all three together]

    So: take traces. A retraction has `trace (l ∘ pl) = trace (pl ∘ l) = trace id`, which is
    the dimension, so each of the two contributes `2` while the identity contributes `2`, and
    `2 + 2 = 2` is false. The obstruction is size and not σ: it holds at `σ = −1`, at
    `0` and at `1` alike, because the count never mentions the multiplication. This is a
    different obstruction from the associativity one in `theory/DEAD_ENDS.md`, which is about
    every carrier rather than about this one, and it is why the file below moves to a carrier
    with infinitely many positions. -/
theorem no_two_wire_split_on_cell :
    ¬ ∃ l r pl pr : Cell →ₗ[ℝ] Cell,
      pl * l = LinearMap.id ∧ pr * r = LinearMap.id ∧
      l * pl + r * pr = LinearMap.id := by
  rintro ⟨l, r, pl, pr, hl, hr, hsum⟩
  have h2 : (LinearMap.trace ℝ Cell) LinearMap.id = (2 : ℝ) := by
    rw [LinearMap.trace_id, Module.finrank_prod, Module.finrank_self]
    norm_num
  have hlt : (LinearMap.trace ℝ Cell) (l * pl) = 2 := by
    rw [LinearMap.trace_mul_comm, hl, h2]
  have hrt : (LinearMap.trace ℝ Cell) (r * pr) = 2 := by
    rw [LinearMap.trace_mul_comm, hr, h2]
  have hsum' := congrArg (LinearMap.trace ℝ Cell) hsum
  rw [map_add, hlt, hrt, h2] at hsum'
  norm_num at hsum'

-- ═══════════════════════════════════════════════════════════════════
-- A CARRIER THAT DOES HOLD TWO COPIES OF ITSELF
-- ═══════════════════════════════════════════════════════════════════

/-- Place a wire in the first of the two tracks.

    (w : ℕ → Cell)                     [a wire carrying one point at each position]
    ℕ → Cell                           [the wire moved into the first track]

    So: the standard pairing sends the first track to the even positions. Every other position
    reads the origin. -/
def leftIn (w : ℕ → Cell) : ℕ → Cell :=
  fun n => Sum.elim w (fun _ => (0 : Cell)) (Equiv.natSumNatEquivNat.symm n)

/-- Place a wire in the second of the two tracks.

    (w : ℕ → Cell)                     [a wire carrying one point at each position]
    ℕ → Cell                           [the wire moved into the second track] -/
def rightIn (w : ℕ → Cell) : ℕ → Cell :=
  fun n => Sum.elim (fun _ => (0 : Cell)) w (Equiv.natSumNatEquivNat.symm n)

/-- Read the first track back off a wire.

    (w : ℕ → Cell)                     [a wire]
    ℕ → Cell                           [what sits in the first track] -/
def leftOut (w : ℕ → Cell) : ℕ → Cell :=
  fun k => w (Equiv.natSumNatEquivNat (Sum.inl k))

/-- Read the second track back off a wire.

    (w : ℕ → Cell)                     [a wire]
    ℕ → Cell                           [what sits in the second track] -/
def rightOut (w : ℕ → Cell) : ℕ → Cell :=
  fun k => w (Equiv.natSumNatEquivNat (Sum.inr k))

/-- Placing in the first track and reading it back changes nothing.

    (w : ℕ → Cell)                     [a wire]
    leftOut (leftIn w) = w             [the round trip is the identity] -/
theorem leftOut_leftIn (w : ℕ → Cell) : leftOut (leftIn w) = w := by
  funext k
  simp [leftOut, leftIn]

/-- Placing in the second track and reading it back changes nothing.

    (w : ℕ → Cell)                     [a wire]
    rightOut (rightIn w) = w           [the round trip is the identity] -/
theorem rightOut_rightIn (w : ℕ → Cell) : rightOut (rightIn w) = w := by
  funext k
  simp [rightOut, rightIn]

/-- The two tracks do not overhear each other.

    (w : ℕ → Cell)                                    [a wire]
    leftOut (rightIn w) = fun _ => (0 : Cell)         [the first track reads nothing]
    rightOut (leftIn w) = fun _ => (0 : Cell)         [and neither does the second]
    ∧                                                 [both together]

    So: the two placements have orthogonal images, which with the two round trips above is what
    Abramsky, Haghverdi and Scott ask of a retraction `U ⊕ U ◁ U`. -/
theorem tracks_orthogonal (w : ℕ → Cell) :
    leftOut (rightIn w) = (fun _ => (0 : Cell)) ∧
      rightOut (leftIn w) = (fun _ => (0 : Cell)) := by
  constructor
  · funext k
    simp [leftOut, rightIn]
  · funext k
    simp [rightOut, leftIn]

/-- The two tracks fill the wire.

    (w : ℕ → Cell)                                              [a wire]
    leftIn (leftOut w) + rightIn (rightOut w) = w               [read both, put both back]

    So: every position lies in one track or the other, so nothing is lost and nothing is
    counted twice. This is the equation `no_two_wire_split_on_cell` shows is unavailable on a
    two-dimensional carrier. -/
theorem in_out_id (w : ℕ → Cell) :
    leftIn (leftOut w) + rightIn (rightOut w) = w := by
  funext n
  simp only [leftIn, rightIn, Pi.add_apply]
  rcases hn : Equiv.natSumNatEquivNat.symm n with k | k
  · have hk : Equiv.natSumNatEquivNat (Sum.inl k) = n := by
      rw [← hn, Equiv.apply_symm_apply]
    simp only [Sum.elim_inl, add_zero]
    show w (Equiv.natSumNatEquivNat (Sum.inl k)) = w n
    rw [hk]
  · have hk : Equiv.natSumNatEquivNat (Sum.inr k) = n := by
      rw [← hn, Equiv.apply_symm_apply]
    simp only [Sum.elim_inr, zero_add]
    show w (Equiv.natSumNatEquivNat (Sum.inr k)) = w n
    rw [hk]

/-- An algebra weight rides along the first track.

    (σ : ℝ) (c : Cell) (w : ℕ → Cell)                       [the parameter, a weight, a wire]
    leftIn (fun n => mul σ c (w n))                              [weight then place]
    = fun n => mul σ c (leftIn w n)                              [is place then weight]

    So: the wires carry the algebra's own multiplication along them, so a feedback sum over
    these wires multiplies weights along a path and adds them across paths, which is the shape
    `Trichotomy/Trace.lean`'s `execution_formula` produces. -/
theorem leftIn_weighted (σ : ℝ) (c : Cell) (w : ℕ → Cell) :
    leftIn (fun n => mul σ c (w n)) = fun n => mul σ c (leftIn w n) := by
  funext n
  simp only [leftIn]
  rcases Equiv.natSumNatEquivNat.symm n with k | k
  · simp
  · simp [mul, Prod.ext_iff]

/-- An algebra weight rides along the second track.

    (σ : ℝ) (c : Cell) (w : ℕ → Cell)                       [the parameter, a weight, a wire]
    rightIn (fun n => mul σ c (w n))                             [weight then place]
    = fun n => mul σ c (rightIn w n)                             [is place then weight] -/
theorem rightIn_weighted (σ : ℝ) (c : Cell) (w : ℕ → Cell) :
    rightIn (fun n => mul σ c (w n)) = fun n => mul σ c (rightIn w n) := by
  funext n
  simp only [rightIn]
  rcases Equiv.natSumNatEquivNat.symm n with k | k
  · simp [mul, Prod.ext_iff]
  · simp

-- ═══════════════════════════════════════════════════════════════════
-- HOW MANY REAL WIRES ONE ALGEBRA WIRE IS
-- ═══════════════════════════════════════════════════════════════════

/-- **One algebra-weighted wire is two real wires exactly above zero.**

    (σ : ℝ)                                                      [the parameter]
    ∃ w : ℕ → Cell                                               [there is a wire]
    (fun n => upper σ (w n)) ≠ (fun n => lower σ (w n))           [whose two readings differ]
    ↔                                                            [exactly when]
    0 < σ                                                        [σ is above zero]

    So: a wire carrying points of the algebra carries two independent real wires on the split
    face and one everywhere else. At or below zero the square root of σ is zero, so the two
    readings of every point agree and the second wire is a copy of the first. This is
    `two_channels_iff_disc_pos` moved onto the carrier, and it is the whole face dependence of
    this file: the retraction and the isometries hold at every setting, the doubling of real
    wires does not. -/
theorem wire_reads_two_reals_iff_disc_pos (σ : ℝ) :
    (∃ w : ℕ → Cell, (fun n => upper σ (w n)) ≠ (fun n => lower σ (w n))) ↔ 0 < σ := by
  constructor
  · rintro ⟨w, hw⟩
    by_cases hσ : 0 < σ
    · exact hσ
    · exact absurd (funext fun n => by
        simp [upper, lower, Real.sqrt_eq_zero_of_nonpos (not_lt.mp hσ)]) hw
  · intro hσ
    refine ⟨fun _ => idemPlus σ, fun h => ?_⟩
    have := congrFun h 0
    rw [(halves_idemPlus hσ).1, (halves_idemPlus hσ).2] at this
    norm_num at this

end Combinator
end Apeiron

end
