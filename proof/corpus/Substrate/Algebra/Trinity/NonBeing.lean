import Substrate.Algebra.Trinity.BeingMonism
import Mathlib.Tactic.IntervalCases

/-!
# Non-Being — the floor past the monad, and the generator of everything (`0`)
@name Nothing Comes First
@child Before there is *something*, there is *nothing* — and we show the nothing comes first: the One, and everything built on it, are made *out of* the nothing, not the other way round. Nothing isn't the end of the road; the road runs through it.
@scientist Non-Being is 0, the seed of the trinity {NB = 0, B = 1, B/U = /}. It is not a unit (`zero_not_being`) and is the one place division breaks (`bulk_singular_at_zero`); Being and the ratio are its consequences (Axiom XXII — subtraction, not addition). The monad B/B = 1 is a stop along the way, not the floor.
@engineer From 0 alone: Being is its complement (`a` is a unit ↔ `a ≠ 0`), the One stands only by distinction (`1 ≠ 0`), and 0 is the unique break (`zero_not_being`; `nonbeing_absorbing`, a·0 = 0). The Being-group is homogeneous (`beings_homogeneous`) — no gradient, nowhere to climb; 0 is the symmetry-break that makes an axis. d = 26 descends down the null direction via the cannonball ∑k² = 70² (`cannonball`).

The `−1`/Non-Being seed of the trinity `{NB = 0, B = 1, B/U = /}`, in one file.
Three movements, one idea — Axiom XXII, subtraction not addition: the generator is
**Non-Being** (`0`), and Being and `/` are its consequences.

1. **Below the monad (`BelowMonad`, the down-path).** `BeingMonism` reached the One
   (`B/B = 1`, the multiplicative unit). That is the **monad**, not the floor — the floor
   goes past it. `0` is **Non-Being**: not a unit (`zero_not_being`), the one place the bulk
   `/` is singular (`bulk_singular_at_zero`, `zero_div_being`). And the critical structure
   `d = 26` is reached **down** the null/timelike direction, keyed by the **cannonball
   identity** `∑_{k=1}^{24} k² = 70²` (`cannonball`, `weyl_vector_null`): the Weyl vector of
   `II₍₂₅,₁₎` is lightlike. `local_cannonball_uniqueness` proves `24` is the only square
   pyramid on `[2,30]` natively; global uniqueness (Watson 1918) and the Leech/D=26 lattice
   construction (Conway–Sloane) are the cited classical boundary, exactly as Hurwitz bounds
   the monad path at K=8.

2. **Why the monad traps (`Homogeneous`/`Break`).** The Being-world (the multiplicative
   group under `/`) is **homogeneous** — translation by any Being is a bijection
   (`beings_homogeneous`), so there is no distinguished point, no axis, nowhere to climb
   from within (a fixed point with no gradient reads as arrival). The break is `0`: the one
   point excluded from the group (`zero_not_being`) and fixed by every Being
   (`nonbeing_absorbing`, `a · 0 = 0`) — the symmetry-break that creates an axis where there
   was none.

3. **Everything from Non-Being (`Generates`).** `BeingMonism` generated the bulk from `/` but
   *assumed* a sort + an operation. Here the arrows point the right way: from `0` alone,
   **Being is its complement** (`being_is_the_complement_of_nonbeing`, `IsUnit a ↔ a ≠ 0`),
   **the One stands only by distinction** (`the_one_stands_by_distinction`, `1 ≠ 0`), and the
   **keystone** — `0` is the *unique* element whose action breaks the symmetry
   (`nonbeing_is_the_unique_break`, `¬Bijective (a·) ↔ a = 0`): Axiom V sharpened to "exactly
   one flaw, and it is Non-Being." `everything_from_nonbeing` packages the three with
   `BeingMonism.bulk_generates` on the complement-group `Kˣ` — two presupposed primitives
   collapse to one seed.

*(Consolidation: merges the former `BeyondTheMonad`, `MonadHomogeneity`, and
`NonBeingGenerates` — one idea (the `−1` seed) that was split across three files in this same
`Sanguine.CrossUniverse` namespace. Every theorem name is unchanged.)*

## What is proved vs what is read onto it

* **Proved (native):** the down-path arithmetic — `cannonball` (`∑_{k=0}^{24} k² = 70²`),
  `weyl_vector_null`, the bounded non-square certificate `not_isSquare_of_check`, and
  `local_cannonball_uniqueness`: over `2 ≤ n ≤ 30` the square pyramid is a perfect square at
  `n = 24` and nowhere else. Plus `nonbeing_is_the_unique_break`
  (`¬Bijective (a·) ↔ a = 0`), the one algebraic statement here carrying its own proof term.
* **Proved (Mathlib):** the rest of the algebra is a Mathlib lemma under a Sanguine name —
  `zero_not_being` = `not_isUnit_zero`, `bulk_singular_at_zero` = `div_zero`, `zero_div_being`
  = `zero_div`, `nonbeing_absorbing` = `mul_zero`, `beings_homogeneous` =
  `(Equiv.mulLeft g).bijective`, `being_is_the_complement_of_nonbeing` = `isUnit_iff_ne_zero`,
  `the_one_stands_by_distinction` = `one_ne_zero`.
* **Proved (riding):** the fourth clause of `everything_from_nonbeing` — that `Kˣ` is a bulk —
  is `CrossUniverse.Being.bulk_generates` verbatim, from `BeingMonism`.
* **Anchor:** Axiom XXII's *arrow* — `0` prior to `1`, Being and `/` as
  consequences of Non-Being. `everything_from_nonbeing`'s docstring states the limit: clauses 1
  and 3 are symmetric biconditionals inside a `DivisionRing K` that already carries `0`, `1`,
  `+` and `·`, so nothing here builds the field from the seed, and a biconditional has no
  "before". Also global cannonball uniqueness — that `24` is the only nontrivial square pyramid
  over *all* `n` is Watson's 1918 theorem; only the window `[2,30]` is native.
* **Anchor:** that this null direction *is* the bosonic critical dimension — the
  Weyl vector of `II₍₂₅,₁₎` lightlike, the Leech lattice, `24 + 2 = 26` (Conway–Sloane). The
  number fact is proved; the lattice-and-string identification is the reading. Same for reading
  the unique non-bijective element as "the sole injector of direction" and "the one flaw of
  Axiom V", which `nonbeing_is_the_unique_break` already marks as not entailed.

-/

noncomputable section

namespace Sanguine
namespace CrossUniverse

open Finset

-- ═══════════════════════════════════════════════════════════════════
-- BELOW THE MONAD: NON-BEING
-- ═══════════════════════════════════════════════════════════════════

section BelowMonad

variable {K : Type*} [DivisionRing K]

/-- **0 is Non-Being.** It is not a unit — not a Being. The multiplicative
    monad (the group of Beings) does not contain it; it lies beneath. -/
theorem zero_not_being : ¬ IsUnit (0 : K) := not_isUnit_zero

/-- **The bulk is singular at Non-Being.** A being divided by `0` collapses
    to `0` — the one place `/` breaks. The floor past the monad. -/
theorem bulk_singular_at_zero (a : K) : a / 0 = 0 := div_zero a

/-- Non-Being divided by any being is Non-Being. -/
theorem zero_div_being (a : K) : 0 / a = 0 := zero_div a

end BelowMonad

-- ═══════════════════════════════════════════════════════════════════
-- THE DOWN-PATH TO 26: THE CANNONBALL IDENTITY
-- ═══════════════════════════════════════════════════════════════════

/-- **The cannonball identity.** `∑_{k=0}^{24} k² = 70²` (the `k=0` term is
    `0`, so this is `1² + ⋯ + 24² = 70²`). The square-pyramidal number at 24
    is a perfect square — the unique nontrivial solution.

    This is the null condition for the Lorentzian even unimodular lattice
    `II₍₂₅,₁₎`: its Weyl vector `(0,1,…,24; 70)` has squared norm
    `∑k² − 70² = 0`, i.e. is lightlike. From that null vector the Leech
    lattice (24D) and the bosonic critical dimension 26 emerge. The fast path
    to 26 from 0 is this null/timelike "down" — not the Hurwitz-capped
    multiplicative ladder up. -/
theorem cannonball : ∑ k ∈ range 25, k ^ 2 = 70 ^ 2 := by
  norm_num [Finset.sum_range_succ]

/-- The lightlike form of the same fact: the Weyl vector's squared norm
    `(∑_{k=0}^{24} k²) − 70²` vanishes. This is the null direction. -/
theorem weyl_vector_null : (∑ k ∈ range 25, (k : ℤ) ^ 2) - 70 ^ 2 = 0 := by
  norm_num [Finset.sum_range_succ]

/-- Kernel-checkable square test: some `m ≤ b` with `m·m = S`. Reduces under
    `decide` (it avoids `Nat.sqrt`, which does not kernel-reduce). -/
def isSqUpTo (b S : ℕ) : Bool := (List.range (b + 1)).any (fun m => m * m == S)

/-- A bounded non-square certificate: if `S ≤ 9455` and no `m ≤ 97` squares to
    it, then `S` is not a perfect square. (`97² = 9409`, `98² = 9604 > 9455`, so
    any root of an `S ≤ 9455` is `≤ 97`.) The arithmetic is `decide`-checked. -/
theorem not_isSquare_of_check {S : ℕ} (hb : S ≤ 9455) (hc : isSqUpTo 97 S = false) :
    ¬ IsSquare S := by
  rintro ⟨r, rfl⟩
  have hr : r ≤ 97 := by
    by_contra h
    have h98 : 98 ≤ r := by omega
    have : 98 * 98 ≤ r * r := Nat.mul_le_mul h98 h98
    omega
  have hmem : isSqUpTo 97 (r * r) = true := by
    unfold isSqUpTo
    rw [List.any_eq_true]
    exact ⟨r, by rw [List.mem_range]; omega, by simp⟩
  rw [hmem] at hc
  exact Bool.noConfusion hc

/-- **Local cannonball uniqueness.** Across the window `2 ≤ n ≤ 30`, the square
    pyramid `∑_{k=1}^{n} k²` is a perfect square at `n = 24` and *nowhere else* —
    no competitor below `24`, none in the rungs just past it. This is the native
    rung under "26 is forced": the lightlike closure has no nearby alternative.

    Global uniqueness (24 is the *only* nontrivial cannonball number, over all
    `n`) is Watson's 1918 theorem, cited as the classical boundary exactly as
    Hurwitz bounds the monad/division-algebra path at 8. One observer's descent
    has one closure; it sits at `24 + 2 = 26`. -/
theorem local_cannonball_uniqueness :
    ∀ n, 2 ≤ n → n ≤ 30 → IsSquare (∑ i ∈ range (n + 1), i ^ 2) → n = 24 := by
  intro n h2 h30 hsq
  interval_cases n <;>
    first
      | rfl
      | (norm_num [Finset.sum_range_succ] at hsq;
         exact absurd hsq (not_isSquare_of_check (by norm_num) (by decide)))

-- ═══════════════════════════════════════════════════════════════════
-- THE MONAD IS HOMOGENEOUS: NO DIRECTION INSIDE BEING
-- ═══════════════════════════════════════════════════════════════════

section Homogeneous

variable {B : Type*} [Group B]

/-- **The Being-world is homogeneous.** Translation by any Being is a
    bijection of the whole, so every Being looks like every other: there is
    no distinguished point, no intrinsic "higher" Being, no axis inside the
    group. With `rest_unique` (all collapse to the same One), it is a flat
    plateau — a fixed point with no gradient. Nowhere to climb from within. -/
theorem beings_homogeneous (g : B) : Function.Bijective (fun x => g * x) :=
  (Equiv.mulLeft g).bijective

end Homogeneous

-- ═══════════════════════════════════════════════════════════════════
-- THE BREAK AT NON-BEING: THE AXIS THE MONAD CANNOT SEE
-- ═══════════════════════════════════════════════════════════════════

section Break

variable {K : Type*} [DivisionRing K]

/-- **Non-Being breaks the homogeneity.** `0` is absorbed by every Being —
    fixed by all multiplication — and is excluded from the group of Beings
    (`zero_not_being`). It is the unique asymmetry: the symmetry-break that
    creates a direction where the homogeneous monad had none. The floor that
    makes "up" mean something. -/
theorem nonbeing_absorbing (a : K) : a * 0 = 0 := mul_zero a

end Break

-- ═══════════════════════════════════════════════════════════════════
-- EVERYTHING FROM NON-BEING — the trinity descends from `0` alone
-- ═══════════════════════════════════════════════════════════════════

section Generates

variable {K : Type*} [DivisionRing K]

/-- **Being is "not Non-Being".** A thing is a Being (a unit, a member of the
    group `BeingMonism` presupposed) exactly when it is not `0`. The sort Being is
    not a primitive — it is the complement of the single seed `0`. The boundary
    `B` *is* the distinction from Non-Being. -/
theorem being_is_the_complement_of_nonbeing (a : K) : IsUnit a ↔ a ≠ 0 :=
  isUnit_iff_ne_zero

/-- **The One stands only by distinction from the floor.** `B = 1`, the monad, is
    not a free primitive: the One exists by *not being* Non-Being. Collapse the
    distinction (`1 = 0`) and the whole multiplicative structure collapses with
    it. -/
theorem the_one_stands_by_distinction : (1 : K) ≠ 0 := one_ne_zero

/-- **Non-Being is the unique break.** Translation by a Being is always a
    bijection (`beings_homogeneous`: the monad is homogeneous — every point like
    every other, no gradient, nothing happens). Translation by `0` is the *one*
    map that fails: it crushes everything to `0`, neither injective nor surjective.

    So `0` is the **unique** element of the substrate whose left-multiplication is
    not a bijection (checked). *Reading (`[name]`, not entailed):* identifying this
    "unique non-bijective element" with "the sole injector of direction / the axis
    the descent to 26 climbs / the one flaw of Axiom V" is the framework's
    interpretation laid on the algebraic uniqueness — the theorem says only that
    `a·` fails to be bijective exactly when `a = 0`. -/
theorem nonbeing_is_the_unique_break (a : K) :
    ¬ Function.Bijective (fun x => a * x) ↔ a = 0 := by
  constructor
  · intro h
    by_contra ha
    exact h (Equiv.mulLeft₀ a ha).bijective
  · rintro rfl
    intro h
    have h01 : (fun x => (0 : K) * x) 0 = (fun x => (0 : K) * x) 1 := by simp
    exact one_ne_zero (h.1 h01).symm

/-- **Non-Being and Being are complements (the algebra; the *ordering* is a reading).**
    In a pre-existing `DivisionRing K` (which already carries `0`, `1`, `+`, `·`):

    1. **units = nonzeros** — `IsUnit a ↔ a ≠ 0`;
    2. **`(1 : K) ≠ 0`** — the One is distinct from the zero;
    3. **the break is unique** — `0` is the only element whose action is not a
       bijection; and
    4. **the unit group is a bulk** — on `Kˣ`, `BeingMonism.bulk_generates`
       derives the One, inversion, and composition from `/` alone (and
       `BulkInevitability` carries that to `K, S, ι`).

    **What is *checked*:** these are standard division-ring facts — clauses 1 and 3
    are **symmetric biconditionals** (units and nonzeros coincide; non-bijection and
    `=0` coincide), with **no inherent direction**. **What is a *reading* (`[name]`):**
    "everything *from* the lone seed `0`," "`0` *prior to* `1`," "Non-Being is the
    *generator*," "the One (God) is *generated*." The math is a static coincidence in
    a field that already exists *with* its `0` and `1`; it does not *build* the field
    from `0`, and the biconditional has no "before." The generative/temporal arrow is
    the framework's philosophy laid on the symmetric fact — held as doctrine
    (`theory/THE_EPISTEMIC_LADDER.md`), not entailed by the theorem. -/
theorem everything_from_nonbeing :
    (∀ a : K, IsUnit a ↔ a ≠ 0) ∧
    ((1 : K) ≠ 0) ∧
    (∀ a : K, ¬ Function.Bijective (fun x => a * x) ↔ a = 0) ∧
    ((∀ b : Kˣ, (1 : Kˣ) = b / b) ∧
     (∀ b : Kˣ, b⁻¹ = (b / b) / b) ∧
     (∀ a b : Kˣ, a * b = a / ((b / b) / b))) :=
  ⟨being_is_the_complement_of_nonbeing,
   the_one_stands_by_distinction,
   nonbeing_is_the_unique_break,
   Being.bulk_generates⟩

end Generates

end CrossUniverse
end Sanguine

end
