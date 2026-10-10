import Apeiron.Decider.Cert

/-!
# The decider, as one fold

@name The Decider
@grade proved

The two phases put together, in the order `../song/src/prover/cone/ncone0.slang`'s `foldLoop`
runs them: propagate to a fixed point, and if nothing crossed, pick a column and project it away,
until the fuel runs out or no column is left worth projecting. `decideLines` is that fold and it
computes, so the corpus in `Apeiron/Decider/Parity.lean` runs it.

## Three verdicts, one of them proved

UNSAT is the verdict that proves theorems, because a law is discharged by refuting its negation,
and `decide_unsat_sound` is its soundness: no assignment satisfies the system. SAT is claimed only
on a run where every elimination was exact, and it is **not** proved here, exactly as
`../song-ref/lean/Song/proof/prover/ConeFold.lean` left it. MAYBE is the honest third and carries
no claim at all. `../song`'s own answer to the SAT side is a constructed witness replayed by
`satcheck.slang`, not a theorem, and this model does not build one.

## Root

| from | here |
|---|---|
| `propUnsat_sound`, itself from `crossing_ofBounds` and `inCell_meet_iff` | `run_unsat_sound` |
| `eliminate_sound`, itself from `Fragment.combo_holds` | `run_unsat_sound` |
| `Wire.satRows_loadRows` (from `Fragment.two_caps_iff_disc_pos`) | `decide_unsat_sound` |

Nothing new about the algebra is proved here. The fold's soundness is the two phases' soundness
composed along an induction on the fuel, and each phase is rooted in the file that states it.

## The refinement gap, stated

This model and the compiled Iota program are two objects. Nothing here says that
`../song/src/prover/cone/ncone0.slang`, once `songc` has lowered it to machine code, computes what
this fold computes. Three things stand between them and none of them is a theorem:

- **Certificate replay.** Every UNSAT the running decider reports is re-derived by
  `certcheck.slang`, whose language and soundness are `Apeiron/Decider/Cert.lean`'s
  `checkCert_sound`. That is the load-bearing link, and it is the one that does not depend on the
  decider being right about anything.
- **The differential.** `../song`'s corpus runs the decider against other solvers.
  `docs/DESIGN.md` section 11 says plainly why that is weaker than it reads: two oracles that have
  both been unsound agreeing is not soundness.
- **The parity corpus here.** `Apeiron/Decider/Parity.lean` checks that this fold and the
  running decider return the same verdict on the same systems, and replays that decider's own
  emitted certificates through `checkCert`. That is evidence about this model, not about the
  compiled program.

Closing the gap needs a semantics for Iota, which is `../song`'s E14, not a theorem here. What
this file narrows is what rests on the gap: with the certificate path proved, an UNSAT is only as
trustworthy as `certcheck`'s own compilation, not as trustworthy as `ncone0`'s.
-/

namespace Apeiron
namespace Decider

open Trichotomy Fragment

variable {n : ℕ}

/-- What the decider answers.

    Verdict                                  [one of three answers]

    So: `UNSAT`, `SAT` and `MAYBE` on ncone0's own standard output, with exit codes 0, 1 and 3. -/
inductive Verdict : Type
  /-- No assignment satisfies the system. The verdict that proves theorems. -/
  | unsat
  /-- Some assignment satisfies it. Claimed on an all-exact run, not proved here. -/
  | sat
  /-- The decider did not settle it. The honest third. -/
  | maybe
  deriving DecidableEq, Repr

/-- How many rows bound a column from below.

    (v : Fin n) (rs : List (Row n))          [a column and a system]
    : ℕ                                      [how many rows have a positive coefficient there] -/
def lowCount (v : Fin n) (rs : List (Row n)) : ℕ := (lowersOf v rs).length

/-- How many rows bound a column from above.

    (v : Fin n) (rs : List (Row n))          [a column and a system]
    : ℕ                                      [how many rows have a negative coefficient there] -/
def upCount (v : Fin n) (rs : List (Row n)) : ℕ := (uppersOf v rs).length

/-- How many pairs at a column would resolve inexactly.

    (v : Fin n) (rs : List (Row n))          [a column and a system]
    : ℕ                                      [how many pairs have both magnitudes above one]

    So: `countInex` in ncone0, which drives the column choice. A pair with a coefficient of one
    on either side resolves exactly over the whole numbers; a pair with neither does not. -/
def inexCount (v : Fin n) (rs : List (Row n)) : ℕ :=
  ((lowersOf v rs).flatMap fun l =>
    (uppersOf v rs).filter fun u => decide (1 < l.coeff v) && decide (1 < -(u.coeff v))).length

/-- Which column to project away next.

    (rs : List (Row n))                      [a system]
    : Option (Fin n)                         [the column chosen, or none if none is worth it]

    So: `pickVarLoop`. A column with no lower rows or no upper rows costs nothing to project and
    is skipped, exactly as ncone0 skips a zero cost. Among the rest, fewest inexact pairs wins,
    then fewest pairs, then the earlier column. -/
def pickVar (rs : List (Row n)) : Option (Fin n) :=
  ((List.finRange n).filter fun v => decide (0 < lowCount v rs * upCount v rs)).foldl
    (fun best v =>
      match best with
      | none => some v
      | some b =>
          if inexCount v rs < inexCount b rs ∨
              (inexCount v rs = inexCount b rs ∧
                lowCount v rs * upCount v rs < lowCount b rs * upCount b rs) then
            some v
          else some b)
    none

/-- Whether projecting a column loses nothing.

    (v : Fin n) (rs : List (Row n))          [a column and a system]
    : Bool                                   [true when every pair resolves exactly]

    So: ncone0's `S_EXACT`, which it clears when a combination's two multipliers both exceed one
    after their common factor is divided out. A run that never clears it may claim SAT; any
    other run may not. -/
def exactAt (v : Fin n) (rs : List (Row n)) : Bool :=
  (lowersOf v rs).all fun l =>
    (uppersOf v rs).all fun u =>
      !(decide (1 < l.coeff v / ((Int.gcd (l.coeff v) (-(u.coeff v)) : ℕ) : ℤ)) &&
        decide (1 < -(u.coeff v) / ((Int.gcd (l.coeff v) (-(u.coeff v)) : ℕ) : ℤ)))

/-- The fold: propagate, then project one column, until the fuel runs out.

    ℕ → Bool → List (Row n) → Verdict        [fuel, exactness so far, the system, the verdict]

    So: `foldLoop`. Propagation runs first at every turn with its own fuel of two hundred, which
    is ncone0's; a crossing there is the verdict. -/
def run {n : ℕ} : ℕ → Bool → List (Row n) → Verdict
  | 0, ex, rs => if propUnsat 200 rs then .unsat else if ex then .sat else .maybe
  | fuel + 1, ex, rs =>
      if propUnsat 200 rs then .unsat
      else
        match pickVar rs with
        | none => if ex then .sat else .maybe
        | some v => run fuel (ex && exactAt v rs) (eliminate v rs)

/-- The decision, on a system as it arrives on the wire.

    (ls : List (Line n))                     [the system, as it arrives on the wire]
    : Verdict                                [the answer]

    So: `oneSystem`, which loads the lines, splits the equalities and calls the fold with a fuel
    of one more than the number of columns. -/
def decideLines (ls : List (Line n)) : Verdict := run (n + 1) true (loadRows ls)

/-- The fold never reports a refutation that is not one.

    ∀ (fuel : ℕ) (ex : Bool) (rs : List (Row n))         [any fuel, exactness flag and system]
    run fuel ex rs = Verdict.unsat                       [the fold reports the system refuted]
    ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs                      [then no assignment satisfies it]

    So: the two phases composed. Propagation reports a crossing only when there was no solution
    to lose, by `propUnsat_sound`, and projection keeps every solution, by `eliminate_sound`, so
    a refutation found after any number of projections is a refutation of the system it started
    from. -/
theorem run_unsat_sound :
    ∀ (fuel : ℕ) (ex : Bool) (rs : List (Row n)),
      run fuel ex rs = Verdict.unsat → ¬ ∃ ρ : Fin n → ℤ, SatRows ρ rs
  | 0, ex, rs, h => by
      rw [run] at h
      split at h
      · next hp => exact propUnsat_sound hp
      · split at h <;> exact absurd h (by simp)
  | fuel + 1, ex, rs, h => by
      rw [run] at h
      split at h
      · next hp => exact propUnsat_sound hp
      · split at h
        · split at h <;> exact absurd h (by simp)
        · next v hv =>
            rintro ⟨ρ, hρ⟩
            exact run_unsat_sound fuel _ _ h ⟨ρ, eliminate_sound hρ v⟩

/-- **The headline: an UNSAT from this fold refutes the wire system.**

    (ls : List (Line n))                                 [a system, as it arrives on the wire]
    decideLines ls = Verdict.unsat                       [the decider answers UNSAT]
    ¬ ∃ ρ : Fin n → ℤ, SatLines ρ ls                     [then no assignment satisfies it]

    So: `../song-ref/lean/Song/proof/prover/ConeFold.lean`'s `decide_unsat_sound` re-derived for
    `ncone0`, on `Apeiron/Fragment/Basic.lean`'s row and the trichotomy's own meet and crossing
    rather than on Lean core's omega constraints. The equality split is where `satRows_loadRows`
    enters, and it is the only place the wire's second line kind is used. -/
theorem decide_unsat_sound (ls : List (Line n)) (h : decideLines ls = Verdict.unsat) :
    ¬ ∃ ρ : Fin n → ℤ, SatLines ρ ls := by
  rintro ⟨ρ, hρ⟩
  exact run_unsat_sound _ _ _ h ⟨ρ, satRows_loadRows hρ⟩

end Decider
end Apeiron
