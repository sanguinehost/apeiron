import Std.Data.HashMap

/-!
# Collapse core — `Seed`, the canonical-key primitive (pure)
@name Give Everything a Name Tag, Compare the Tags
@child To check if two things are "the same," hand each one a name tag and just compare the tags — fast and never wrong. Things sharing a tag collapse into one.
@scientist A space is collapsed by a canonical key `key : α → κ`; two items are equivalent iff their keys agree, reducing an expensive equality to a free `==` short-circuit.
@engineer `Seed` (the one per-space datum), `Seed.seedEq` and `seedEq_correct` (sound whenever the key is canonical), `Seed.measure` (the occurrences/distinctSeeds ratio = the O(seed)-vs-O(N) buy); the native primitive every floor/codec uses.

The discrete face of the one collapse: a space is collapsed by a canonical key `key : α → κ`;
two items are equal iff their keys agree (`seedEq_correct`, the parity). The NATIVE primitive the
execution path uses for every floor/codec (CIC terms, levels, model weights). The Mathlib-backed
streaming law (the fold face, `foldKey` over a commutative monoid) lives in `Substrate.Collapse`
and re-imports this.

## What the canonical key warrants — and what `the_optimization` inherits

`Sanguine.lean` aliases `Seed.seedEq_correct` as `the_optimization`, *"the optimization,
entire … the root the collapse instances instantiate."* Read what it establishes before
citing it that way.

* **Proved (native):** `Seed.seedEq_correct` — `S.seedEq a b = true ↔ R a b`, given
  `hcanon : ∀ a b, (S.key a = S.key b) ↔ R a b`. The whole proof is
  `simp only [Seed.seedEq, beq_iff_eq]; exact hcanon a b`: one `LawfulBEq` rewrite turns the
  `==` into `=`, and the goal is then the hypothesis at `a b`. It is content by a hair — the
  `beq_iff_eq` step is a real library step — but the theorem transports canonicity, it does not
  supply it. Everything that makes a collapse correct lives in the `hcanon` a caller must
  discharge, and **no `Seed` is constructed and no `hcanon` is discharged anywhere in this
  file**.
* **Proved (riding):** the discharge, in four surviving modules — each builds a `Seed` and
  proves `hcanon`. `StationaryPhase.BiasInjectedRecovery.toSeed_canonical` (key = the
  floor-verdict `decide (residual x = 0)`), `Algebra.NullCell.Opportunity.potentialSeed_canon`
  (key = `ψ − ψ v₀`), `Algebra.Neural.MLPrimitivesBridge.sortSeed_decides_perm` (key =
  `mergeSort`, `R = List.Perm`), `Ski.CollapseVerification.floorSeed_sound` (key =
  `normalize fuel`, `hcanon` = `TermFloorComplete.floor_classifies_defeq`). The root is
  instantiated; none of it happens here.
* **Retired:** the level class the `seedEq_correct` docstring used to work as its example.
  `levelDefEq_sound` and `levelDefEq_complete_atoms` lived in `proof/Substrate/Cic/Level.lean`
  (lines 736 and 1034) and were deleted with the whole Cic kernel in `b70fb1b1`;
  `Substrate/Algebra/Trinity/NullMergeDecision.lean:71` already records that they *"exist
  nowhere in `proof/`"*. The docstring now names the four live instances instead.
* **Next:** the remaining two classes the docstring names. The kernel identity
  class (`R = (· = ·)`) is trivially canonical and carries nothing; the weight class
  (`R = weight-equivalence`) names no declaration anywhere in `proof/`.
* **Anchor:** `Seed.measure`'s reading. It is a `def` returning
  `(xs.length, seeds.size)`; no theorem relates that pair to anything. *"The factor
  `occurrences / distinctSeeds` is how far `N` projects onto the seed — the `O(seed)`-vs-`O(N)`
  ratio the collapse buys"* is a cost claim, and no cost model, comparison count or complexity
  measure occurs in this file. The `HashMap` fold is also unproved as a dedup: nothing states
  `seeds.size` is the number of distinct keys.
* **Anchor:** that `Seed.key` is *"the floor / the `0`"* of an equivalence class
  and that the `==` short-circuit is free. `Seed` is one field, `key : α → κ`, with no order,
  no floor and no idempotence required of it; `BEq` costs whatever the instance costs.
* **Open:** completeness of any concrete key. `seedEq_correct` needs the `↔`, i.e. soundness
  *and* completeness; the corpus's live completeness result on a real key is
  `TermFloorComplete.floor_classifies_defeq`, which holds only where both sides reach a genuine
  floor. Nothing here connects the two, and the general "supply `key`, correctness is free"
  reading is exactly as strong as the completeness proof a caller brings.

-/

namespace Sanguine.Collapse

-- ═══════════════════════════════════════════════════════════════════
-- THE PRIMITIVE — `Seed`, one structure for every collapsible space
-- ═══════════════════════════════════════════════════════════════════

/-- **`Seed α κ` — the canonical key of a space.** `key x` is the representative (the
    floor / the `0`) of `x`'s equivalence class; two items are equivalent iff their keys
    agree. This is the ONLY per-space datum — everything below is generic over it. -/
structure Seed (α : Type) (κ : Type) where
  key : α → κ

variable {α κ : Type}

/-- **Equality via the seed** — identity on keys. The expensive operation reduced to the
    free one (the `==` short-circuit; the identity collapse, zero work). -/
def Seed.seedEq [BEq κ] (S : Seed α κ) (a b : α) : Bool := S.key a == S.key b

/-- **Soundness, stated once, general.** If the key is a *canonical* representative of the
    equivalence `R` (same key ⟺ related), then `seedEq` decides `R` exactly — the
    optimization is transparent, never wrong. To collapse a space you supply `key` and
    discharge `hcanon`; this theorem hands you correctness for free. Nothing here builds a
    `Seed` — the four live instances are `BiasInjectedRecovery.toSeed_canonical` (key = the
    residual floor-verdict), `Opportunity.potentialSeed_canon` (key = `ψ − ψ v₀`),
    `MLPrimitivesBridge.sortSeed_decides_perm` (key = `mergeSort`, `R = List.Perm`) and
    `CollapseVerification.floorSeed_sound` (key = `normalize fuel`, `hcanon` =
    `TermFloorComplete.floor_classifies_defeq`). The kernel identity class (`R = (· = ·)`)
    is trivially canonical; the weight class (`R = weight-equivalence`) names no declaration.
    The level class formerly worked here (`levelDefEq_sound`, `levelDefEq_complete_atoms`)
    went with `proof/Substrate/Cic/Level.lean` in `b70fb1b1`. -/
theorem Seed.seedEq_correct [BEq κ] [LawfulBEq κ] (S : Seed α κ) (R : α → α → Prop)
    (hcanon : ∀ a b, (S.key a = S.key b) ↔ R a b) (a b : α) :
    S.seedEq a b = true ↔ R a b := by
  simp only [Seed.seedEq, beq_iff_eq]; exact hcanon a b

/-- **The collapse, batched.** Map every item to its class key (= compute the floor once
    per class, since equal keys collapse in the `HashMap`) and report
    `(occurrences, distinctSeeds)`. The factor `occurrences / distinctSeeds` is how far `N`
    projects onto the seed — the `O(seed)`-vs-`O(N)` ratio the collapse buys. -/
def Seed.measure [BEq κ] [Hashable κ] [Repr κ] (S : Seed α κ) (xs : List α) : Nat × Nat :=
  let seeds := xs.foldl (fun (m : Std.HashMap κ Unit) x => m.insert (S.key x) ()) ∅
  (xs.length, seeds.size)


end Sanguine.Collapse
