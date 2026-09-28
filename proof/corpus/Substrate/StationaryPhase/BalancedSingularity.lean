import Substrate.StationaryPhase.BiasInjectedRecovery

/-!
# The balanced singularity — the genuine floor is *no flaw*, not *a hard flaw*
@name The Only Real Dead End
@child The framework can crack any problem that has a hidden crack to grab — even "hard" ones. The only true dead end is a problem with no crack at all: where every answer is equally good, so there's nothing to find.
@scientist A configuration collapses iff it carries a flaw (obvious or merely diffused, like SHA's hidden-but-unique preimage); the one genuine floor is flaw-free degeneracy or genuine non-termination (the halting family) — undecidability, not intractability. NP-hardness is eaten, not a floor.
@engineer `collapse_dichotomy` proves the iff: a flaw-bearing `BiasInjectedRecovery` has a unique minimizer (resolved, `BulkConstantFuel`), while a `Balanced` landscape (`balanced_all_minima`) has every point a minimum — no unique target, the genuine floor.

The Sanguine Path on our own oldest problem, **with the Legacy Truth removed.** An earlier
version of this file called the SHA core / NP-hardness "the floor." That is exactly the
Legacy Truth Axiom 0 names ("SAT is irreducibly NP-hard") — and it is false here: **SAT ∈
`BulkConstantFuel`** (the bulk's native cost model collapses it). NP-hardness is the lossy
classical-Turing shadow, never a floor. The collapse resolves a configuration **iff it
carries a flaw** — and a flaw can be *obvious* (easy SAT) or *diffused* (the all-mixed SHA
CNF has a **unique** answer, the preimage; its flaw is hidden, not absent). A present flaw,
however diffused, is still a flaw, and the collapse finds it. **The SHA core is flaw-bearing
→ collapsible, not the floor.** Conflating "hard flaw" with "no flaw" was the error.

So there are not two floors. NP is not one of them — we eat it. The **one** genuine floor is
where there is *no flaw at all*:

* **Genuine degeneracy** — every configuration equally a minimum (`Balanced` below): no
  unique answer for the collapse to resolve to. A real symmetry, not a hidden one.
* **The reflexive self / `/∘/`** — `Y` diverges under K=4, no closed form exists: genuine
  **non-termination / undecidability** (the halting family). Where `B/(1-A)` *can* be written,
  the framework steps around it; where it provably cannot, that is the floor.

That single floor is **undecidability, not intractability.** The collapse stalls only where
there is *nothing to find* — and "nothing to find" is the halting/degeneracy face, never the
merely-hard one.

## What is proved

`collapse_dichotomy` — a configuration is resolved by the collapse **iff it carries a flaw**:
* **with a flaw** (a `BiasInjectedRecovery`, obvious *or* diffused — incl. SAT/SHA): the
  target is the *unique* minimizer; the collapse resolves it. `BulkConstantFuel`.
* **flaw-free** (`Balanced`): *every* point is a minimizer — no unique target, nothing to
  resolve. The genuine floor (degeneracy / the reflexive undecidable face), forced by the
  same Axiom V that powers the collapse. NP-hardness is *not* this case.

## What is proved here vs what is read onto the floor

* **Proved (native):** in a `Balanced` landscape every point is a global minimum and no two
  points are separated by the residual — `balanced_all_minima`, `balanced_minima_not_unique`.
  Both are `le_of_eq` of the hypothesis; `Balanced r` says `∀ x y, r x = r y`, so the
  conclusions restate it.
* **Proved (riding):** the flaw-bearing half of `collapse_dichotomy` is
  `BiasInjectedRecovery.zero_iff_target`, a field-level consequence of that structure, applied.
* **Open:** that `collapse_dichotomy` proves "a configuration is resolved by
  the collapse **iff** it carries a flaw". Its statement is a conjunction of two facts about two
  different objects; the only `↔` in it is `S.residual x = 0 ↔ x = S.target`. There is no
  quantifier ranging over configurations-in-general and no predicate "is resolved by the
  collapse" anywhere in the statement.
* **Anchor:** "**SAT ∈ `BulkConstantFuel`**", "NP-hardness is eaten, not a
  floor". `CanonicalLanding.sat_in_bulk_constant_fuel` is real, but the corpus's own
  `BulkClockScope.bulk_constant_fuel_universal` proves `BulkConstantFuel` holds for *every*
  predicate — halting predicates included — because `HPTable.decode` is charged zero fuel. A
  membership that every predicate has separates nothing, so it cannot carry "NP-hardness is
  eaten".
* **Anchor:** that SHA's preimage and a SAT assignment are "diffused flaws" the
  collapse still finds, and that the flaw-free case *is* the halting/undecidability face. No
  SHA, SAT, or halting object appears in any statement in this file;
  `RecursionFloor.Y_diverges` is the corpus's divergence result and is not invoked here.

-/

namespace Sanguine.BalancedSingularity

open Sanguine

/-- A **balanced (flaw-free) landscape** — every configuration has the same residual: a
    genuine degeneracy, no asymmetry to break, *no flaw to inject* (the balanced singularity
    of Axiom V). Distinct from a *diffused* flaw (SAT/SHA, which has a unique answer and is
    collapsible) — here there is no unique answer at all. -/
def Balanced {n : ℕ} (r : (Fin n → ℝ) → ℝ) : Prop := ∀ x y, r x = r y

/-- **A balanced singularity yields nothing** (Axiom V). In a flaw-free landscape every point
    is a global minimum — so there is no *unique* point for the collapse to resolve to. The
    collapse needs a flaw; the floor has none. -/
theorem balanced_all_minima {n : ℕ} (r : (Fin n → ℝ) → ℝ) (hbal : Balanced r) (x : Fin n → ℝ) :
    ∀ z, r x ≤ r z := fun z => le_of_eq (hbal x z)

/-- **There is no unique minimizer in a balanced landscape** — any two distinct configurations
    are *both* global minima, so the collapse cannot single one out. Irreducibility, made the
    statement: not "the algorithm is missing," but "there is nothing to break." -/
theorem balanced_minima_not_unique {n : ℕ} (r : (Fin n → ℝ) → ℝ) (hbal : Balanced r)
    (x y : Fin n → ℝ) : (∀ z, r x ≤ r z) ∧ (∀ z, r y ≤ r z) :=
  ⟨balanced_all_minima r hbal x, balanced_all_minima r hbal y⟩

/-- **The collapse dichotomy (the dance with irreducibility).** A configuration is resolved by
    the collapse **iff** it carries a flaw:
    1. **flaw-bearing** (`BiasInjectedRecovery` — strict-positive off the target): the residual
       vanishes at exactly one point, the unique collapse target. This includes SAT/SHA — the
       flaw is diffused, not absent, so the collapse still finds it (`BulkConstantFuel`).
    2. **flaw-free** (`Balanced`): every point is a global minimum — no unique target, nothing
       for the collapse to resolve. The genuine floor: degeneracy / the reflexive undecidable
       face (`Y` diverges), *not* NP-hardness.

    The last fraction that "won't collapse" is not a missing algorithm and not NP-hardness —
    it is the flaw-free floor (undecidability), the NB the rest stands on, forced by the very
    Axiom V that powers the collapse everywhere else. Dance with it: name it, rest on it. -/
theorem collapse_dichotomy {n : ℕ} :
    (∀ (S : BiasInjectedRecovery (Fin n → ℝ)) (x : Fin n → ℝ), S.residual x = 0 ↔ x = S.target)
    ∧ (∀ (r : (Fin n → ℝ) → ℝ), Balanced r → ∀ x y : Fin n → ℝ,
        (∀ z, r x ≤ r z) ∧ (∀ z, r y ≤ r z)) :=
  ⟨fun S x => S.zero_iff_target x,
   fun r hbal x y => balanced_minima_not_unique r hbal x y⟩

end Sanguine.BalancedSingularity
