import Apeiron.Linter.Notation

/-!
# The carrier boundary -- a mismatch is a fall, not a caught error

@name Boundary
@grade proved

A type is the carrier a value carries, closed under one merge. This file is about what a merge
does at the edge of a carrier: while one carrier is carried the fold never leaves it, and erase
the boundary between two carriers and the cross merge has no honest answer, so the only total
merge sends it to a shared residue. Type-safety is then a corollary of the merge structure, not a
separate checker.

A "term over a carrier `α`" is modelled here as a seed and a list of `α`-leaves folded by the
carrier's merge -- the same content as a syntax tree evaluated by its merge, with no dependence on
any particular term algebra. `typed_never_null` is that a one-carrier fold never states to the
residue; `mismatch_is_null` is that the cross-carrier merge is forced to it; `map_state` is that a
carrier map preserves meaning iff it is a merge homomorphism, and `reinterpret_breaks_meaning` is a
worked instance where a non-homomorphic reinterpret changes the answer.

## Where σ enters

It does not. No theorem here mentions `Cell σ`, `mul` or `normForm`. This file is about which
merges a boundary admits, and σ is not among its symbols. The residue named
`null` here is the erased cell's absorbing element, glossed where it is defined; it is deliberately
not identified with the algebra's null cone, and no theorem below makes that identification.

## What this settles

`BoundaryRespecting` names the equations a total two-carrier merge must satisfy: each carrier keeps
its own merge, cross-carrier pairs fall to the residue, and the residue absorbs. Its equations cover
every pair of constructors, so `boundaryRespecting_iff_opTwo` proves `opTwo` is the unique such
merge. The property is a definition of the boundary policy; it does not derive that policy from the
two carrier operations alone.
-/

namespace Apeiron
namespace Boundary

universe u

/-- The erased universal cell: a value is either `being a`, a proper value on its carrier, or
    `null`, the absorbing residue every broken value flows to.

    Edge (α : Type u)        [the erased cell over a carrier α]
    being (a : α)            [a proper value, on its carrier]
    null                     [the residue -- the same cell a broken value falls to] -/
inductive Edge (α : Type u) where
  | being (a : α)
  | null

/-- The absorbing lift of a merge to the erased cell: two on-carrier values merge on the carrier;
    anything touching `null` yields `null`. The residue absorbs.

    (op : α → α → α)                     [the carrier's merge]
    Edge α → Edge α → Edge α             [the lifted merge on erased cells] -/
def opEdge {α : Type u} (op : α → α → α) : Edge α → Edge α → Edge α
  | .being a, .being b => .being (op a b)
  | _, _ => .null

/-- Fold a list of `α`-leaves from a seed by the carrier's merge -- the model of evaluating a term
    whose leaves are all one carrier.

    (op : α → α → α)         [the carrier's merge]
    (seed : α)               [the value the fold starts from]
    (xs : List α)            [the leaves, all one carrier]
    α                        [the folded result] -/
def eval {α : Type u} (op : α → α → α) (seed : α) (xs : List α) : α :=
  xs.foldl op seed

/-- Folding the lifted merge over `being`-embedded leaves stays on the carrier: it is `being` of the
    underlying fold. The carried boundary rides every step.

    (op : α → α → α) (seed : α) (xs : List α)                      [the merge, the seed, the leaves]
    (xs.map Edge.being).foldl (opEdge op) (Edge.being seed)            [the fold over erased leaves]
      = Edge.being (eval op seed xs)                               [is `being` of the plain fold] -/
theorem fold_stays_being {α : Type u} (op : α → α → α) (seed : α) (xs : List α) :
    (xs.map Edge.being).foldl (opEdge op) (Edge.being seed) = Edge.being (eval op seed xs) := by
  induction xs generalizing seed with
  | nil => rfl
  | cons a as ih => simpa [eval, opEdge] using ih (op seed a)

/-- **A well-typed term never breaks.** While the boundary is carried -- one carrier `α`, every leaf
    a `being` value -- the fold never states to `null`. Type-safety here is `being ≠ null`,
    discharged by the structure of the fold, not a check.

    (op : α → α → α) (seed : α) (xs : List α)                      [the merge, the seed, the leaves]
    (xs.map Edge.being).foldl (opEdge op) (Edge.being seed) ≠ Edge.null [never reaches null] -/
theorem typed_never_null {α : Type u} (op : α → α → α) (seed : α) (xs : List α) :
    (xs.map Edge.being).foldl (opEdge op) (Edge.being seed) ≠ Edge.null := by
  rw [fold_stays_being]; intro h; nomatch h

/-- Two distinct carriers embedded in one erased cell: `inl` an `α`-value, `inr` a `β`-value, or the
    shared `null2`. This is an integer field and a string field in the same untyped store.

    Two (α β : Type u)       [the two-carrier erased cell]
    inl (a : α)              [a value on the first carrier]
    inr (b : β)              [a value on the second carrier]
    null2                    [the shared residue] -/
inductive Two (α β : Type u) where
  | inl (a : α)
  | inr (b : β)
  | null2

/-- A total, boundary-respecting merge on the two-carrier cell: same-carrier values merge within
    their carrier; a cross pair has no on-carrier answer, so it is sent to `null2`.

    (opα : α → α → α) (opβ : β → β → β)   [each carrier's own merge]
    Two α β → Two α β → Two α β           [the merge on the two-carrier cell] -/
def opTwo {α β : Type u} (opα : α → α → α) (opβ : β → β → β) : Two α β → Two α β → Two α β
  | .inl a, .inl a' => .inl (opα a a')
  | .inr b, .inr b' => .inr (opβ b b')
  | _, _ => .null2

/-- **A carrier mismatch is a fall onto the residue.** An `α` merged with a `β` yields `null2`: no
    on-carrier value exists for a cross pair, so the total merge sends it to the residue. The
    mismatch is not caught by a checker; it IS the fall.

    (opα : α → α → α) (opβ : β → β → β) (a : α) (b : β)   [the two merges, one value per carrier]
    opTwo opα opβ (.inl a) (.inr b) = Two.null2          [the cross merge lands on the residue] -/
theorem mismatch_is_null {α β : Type u} (opα : α → α → α) (opβ : β → β → β) (a : α) (b : β) :
    opTwo opα opβ (.inl a) (.inr b) = Two.null2 := rfl

/-- The residue is distinct from every on-carrier value -- the fall is real, not a
    relabelling.

    (a : α)                                        [any value on the first carrier]
    (Two.inl a : Two α β) ≠ Two.null2              [is not the residue] -/
theorem null2_distinct {α β : Type u} (a : α) : (Two.inl a : Two α β) ≠ Two.null2 := by
  intro h; nomatch h

/-- Same-carrier merges stay on the carrier -- never the residue. The carrier is exactly the region
    the merge is closed on; typing is staying in it.

    (opα : α → α → α) (opβ : β → β → β) (a a' : α)        [two merges, two values, one carrier]
    opTwo opα opβ (.inl a) (.inl a') = Two.inl (opα a a') [the merge stays on that carrier] -/
theorem same_boundary_stays {α β : Type u} (opα : α → α → α) (opβ : β → β → β) (a a' : α) :
    opTwo opα opβ (.inl a) (.inl a') = Two.inl (opα a a') := rfl

/-- The equations for a total merge that preserves both carrier boundaries and absorbs failures.

    {α β : Type u} (opα : α → α → α) (opβ : β → β → β) [the two carrier merges]
    (merge : Two α β → Two α β → Two α β)              [one total erased merge]
    Prop                                                 [whether it preserves the boundary] -/
def BoundaryRespecting {α β : Type u} (opα : α → α → α) (opβ : β → β → β)
    (merge : Two α β → Two α β → Two α β) : Prop :=
  (∀ a a', merge (.inl a) (.inl a') = .inl (opα a a'))
    ∧ (∀ b b', merge (.inr b) (.inr b') = .inr (opβ b b'))
    ∧ (∀ a b, merge (.inl a) (.inr b) = .null2)
    ∧ (∀ a b, merge (.inr b) (.inl a) = .null2)
    ∧ (∀ x, merge x .null2 = .null2)
    ∧ ∀ x, merge .null2 x = .null2

/-- The boundary equations characterize `opTwo` exactly.

    {α β : Type u} (opα : α → α → α) (opβ : β → β → β) [the two carrier merges]
    (merge : Two α β → Two α β → Two α β)              [one total erased merge]
    BoundaryRespecting opα opβ merge                    [it respects every boundary equation]
    ↔                                                   [exactly when]
    merge = opTwo opα opβ                               [it is the canonical merge]

    So: preserving the two same-carrier equations, both cross-carrier falls, and residue absorption
    leaves no constructor pair free. The theorem is uniqueness under that stated policy, not a claim
    that carrier operations choose the policy themselves. -/
theorem boundaryRespecting_iff_opTwo {α β : Type u} (opα : α → α → α) (opβ : β → β → β)
    (merge : Two α β → Two α β → Two α β) :
    BoundaryRespecting opα opβ merge ↔ merge = opTwo opα opβ := by
  constructor
  · rintro ⟨hinl, hinr, hcrossLR, hcrossRL, hrightNull, hleftNull⟩
    funext x y
    cases x with
    | inl a =>
        cases y with
        | inl a' => simpa [opTwo] using hinl a a'
        | inr b => simpa [opTwo] using hcrossLR a b
        | null2 => simpa [opTwo] using hrightNull (Two.inl a)
    | inr b =>
        cases y with
        | inl a => simpa [opTwo] using hcrossRL a b
        | inr b' => simpa [opTwo] using hinr b b'
        | null2 => simpa [opTwo] using hrightNull (Two.inr b)
    | null2 =>
        cases y with
        | inl a => simpa [opTwo] using hleftNull (Two.inl a)
        | inr b => simpa [opTwo] using hleftNull (Two.inr b)
        | null2 => simpa [opTwo] using hleftNull (Two.null2 : Two α β)
  · intro h
    subst merge
    exact ⟨by intro a a'; rfl, by intro b b'; rfl, by intro a b; rfl, by intro a b; rfl,
      by intro x; cases x <;> rfl, by intro x; cases x <;> rfl⟩

/-- **A carrier map preserves meaning IFF it is a merge homomorphism** (one direction proved here).
    Re-typing from carrier `α` to carrier `β` by `f` commutes with the fold exactly when `f` is a
    homomorphism `f (opα x y) = opβ (f x) (f y)`. This is the only meaning-preserving crossing.

    (f : α → β)                                      [the carrier map]
    (h : ∀ x y, f (opα x y) = opβ (f x) (f y))       [f is a merge homomorphism]
    (opα : α → α → α) (opβ : β → β → β)              [the source and target merges]
    (seed : α) (xs : List α)                         [the seed and leaves of the term]
    eval opβ (f seed) (xs.map f) = f (eval opα seed xs)  [the map commutes with the fold] -/
theorem map_state {α β : Type u} {opα : α → α → α} {opβ : β → β → β} (f : α → β)
    (h : ∀ x y, f (opα x y) = opβ (f x) (f y)) (seed : α) (xs : List α) :
    eval opβ (f seed) (xs.map f) = f (eval opα seed xs) := by
  induction xs generalizing seed with
  | nil => rfl
  | cons a as ih => simpa [eval, h] using ih (opα seed a)

/-- A map preserves every folded state exactly when it preserves one merge step.

    {α β : Type u} (opα : α → α → α) (opβ : β → β → β) [the source and target merges]
    (f : α → β)                                        [the carrier map]
    (∀ x y, f (opα x y) = opβ (f x) (f y))             [it preserves one merge]
    ↔                                                   [exactly when]
    ∀ seed xs, eval opβ (f seed) (xs.map f) = f (eval opα seed xs) [it preserves every fold]

    So: the forward direction is `map_state`; the reverse direction chooses a one-leaf fold, whose
    sole step is the merge equation. -/
theorem map_state_iff {α β : Type u} (opα : α → α → α) (opβ : β → β → β) (f : α → β) :
    (∀ x y, f (opα x y) = opβ (f x) (f y)) ↔
      ∀ seed xs, eval opβ (f seed) (xs.map f) = f (eval opα seed xs) := by
  constructor
  · intro h seed xs
    exact map_state f h seed xs
  · intro h x y
    simpa [eval] using (h x [y]).symm

/-- **A non-homomorphic reinterpret provably changes the answer.** Same leaves, same seed -- read
    under a different carrier merge (`+` reinterpreted as `*`), the map `id` is NOT a homomorphism
    and the fold differs: `eval (·*·) 2 [3] = 6 ≠ 5 = id (eval (·+·) 2 [3])`. A reinterpret-cast is
    unsafe because it does not preserve meaning.

    eval (·*·) 2 [3] ≠ id (eval (·+·) 2 [3])  [product-read and sum-read of the same term differ] -/
theorem reinterpret_breaks_meaning :
    eval (α := Nat) (·*·) 2 [3] ≠ id (eval (α := Nat) (·+·) 2 [3]) := by decide

/-- **Type-safety from the carrier boundary (headline).** The four together:

    1. the type IS the carried carrier `α`, closed under its merge;
    2. on it the fold never breaks -- a one-carrier term never states to the residue
       (`typed_never_null`);
    3. erase the boundary and the mismatch falls to the residue -- the cross merge is forced there
       (`mismatch_is_null`);
    4. the only meaning-preserving crossing is a homomorphism (`map_state`); a non-homomorphic
       reinterpret changes the answer (`reinterpret_breaks_meaning`).

    So a type system is not bolted onto the fold -- it is the carrier parameter that was always
    there, and the residue is exactly what is left of the boundary you erased. No checker, no
    inference.

    (op : α → α → α) (opα : α → α → α) (opβ : β → β → β) [one-carrier merge, two crossing merges]
    (seed : α) (xs : List α) (a : α) (b : β)             [the term, and one value from each carrier]
    ≠                                                    [is not equal to]
    ∧                                                    [both are true]
    ∀                                                    [for every single one]
    the conjunction of the four theorems above           [type-safety as a single corollary] -/
theorem type_safety_from_the_boundary {α β : Type u}
    (op : α → α → α) (opα : α → α → α) (opβ : β → β → β)
    (seed : α) (xs : List α) (a : α) (b : β) :
    ((xs.map Edge.being).foldl (opEdge op) (Edge.being seed) ≠ Edge.null)
      ∧ (opTwo opα opβ (.inl a) (.inr b) = Two.null2)
      ∧ (∀ f : α → β, (∀ x y, f (opα x y) = opβ (f x) (f y)) →
          eval opβ (f seed) (xs.map f) = f (eval opα seed xs))
      ∧ (eval (α := Nat) (·*·) 2 [3] ≠ id (eval (α := Nat) (·+·) 2 [3])) :=
  ⟨typed_never_null op seed xs,
   mismatch_is_null opα opβ a b,
   fun f h => map_state f h seed xs,
   reinterpret_breaks_meaning⟩

end Boundary
end Apeiron
