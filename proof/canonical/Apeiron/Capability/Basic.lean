import Apeiron.Linter.Notation

/-!
# Capability provenance

@name Provenance
@grade proved

A capability presented at a boundary is one that was handed over, never one that was computed.
That is a property of the language a program is written in, not of the bits the capability
happens to carry.

Two sorts. `NExpr` is ordinary arithmetic, where offsets and sizes and loop bounds are computed.
`CExpr` is capability-valued and has exactly two constructors: read one already in scope
(`var`), or take one the compiler minted (`mint`).

**What is absent is the theorem.** There is no constructor taking an offset and a capability and
returning a capability, and no eliminator handing a capability's fields back as arithmetic for
recombination. `reachable_iff_inScope` says the capabilities a term can produce are exactly
those already in scope, and the proof is a case split with nothing left to discharge. Add a
third constructor and the proof breaks at that split, which is the point: the grammar carries
the guarantee, so there is no separate assumption to make.

The two sorts never meet, and `NExpr` is here to make that visible. Nothing below mentions it.

## Where σ enters

It does not, and no attempt is made to bring it in. This file is about which constructors a
grammar has. No theorem here mentions `Cell σ`, `mul` or `normForm`, and one that did would be
decoration. The corpus's other closed calculus, `Combinator.Comb`, is worth the comparison:
`Comb` has `app`, which is why combinatory completeness takes a proof, and `CExpr` has no
combining constructor at all, which is why provenance takes none. The absence is the property.

## What this does not settle

That a real compiler's grammar has this shape is a claim about that compiler, checkable by
reading its parser, and not established here.
-/

namespace Apeiron
namespace Capability

/-- A capability in flight: an index into a grant table, and a tag.

    Cap                  [a capability]
    index : Nat          [which grant it names]
    tag : Nat            [what authenticates it] -/
structure Cap where
  /-- Which grant it names. -/
  index : Nat
  /-- What authenticates it. -/
  tag : Nat
  deriving DecidableEq

/-- The grants handed out, in order.

    Grants Region        [the list of regions granted] -/
abbrev Grants (Region : Type) := List Region

/-- Resolve an index against the grant table.

    {Region : Type}                        [what a region is, left abstract]
    Grants Region → Nat → Option Region    [takes the table and an index, gives a region or none] -/
def lookup {Region : Type} : Grants Region → Nat → Option Region
  | [], _ => none
  | r :: _, 0 => some r
  | _ :: rest, (n + 1) => lookup rest n

/-- Anything an index resolves to was granted.

    {Region : Type}                              [what a region is]
    ∀ (g : Grants Region) (i : Nat) (r : Region) [for every single table, index and region]
    lookup g i = some r → r ∈ g                  [if it resolves there, it was granted] -/
theorem lookup_mem {Region : Type} :
    ∀ (g : Grants Region) (i : Nat) (r : Region), lookup g i = some r → r ∈ g
  | [], _, _, h => by simp [lookup] at h
  | x :: _, 0, r, h => by
      simp [lookup] at h
      exact h ▸ List.Mem.head _
  | _ :: rest, (n + 1), r, h => List.Mem.tail _ (lookup_mem rest n r h)

/-- Arithmetic. The sort a program computes offsets and sizes in, which never mentions `Cap`.

    NExpr                [an arithmetic expression] -/
inductive NExpr : Type
  /-- A literal. -/
  | lit (n : Nat)
  /-- A sum. -/
  | add (a b : NExpr)
  /-- A difference. -/
  | sub (a b : NExpr)
  /-- A product. -/
  | mul (a b : NExpr)

/-- What an arithmetic expression works out to.

    NExpr → Nat          [takes an expression, gives a number] -/
def NExpr.eval : NExpr → Nat
  | .lit n => n
  | .add a b => a.eval + b.eval
  | .sub a b => a.eval - b.eval
  | .mul a b => a.eval * b.eval

/-- Capability-valued expressions, and the whole grammar.

    CExpr                [a capability-valued expression]
    var (i : Nat)        [read one already in scope]
    mint (i : Nat)       [take one the compiler minted]

    So: two constructors and no third. Nothing takes an offset and a capability and returns a
    capability, and nothing reads a capability's fields back out as arithmetic. -/
inductive CExpr : Type
  /-- Read a capability already in scope. -/
  | var (i : Nat)
  /-- Take one the compiler minted. -/
  | mint (i : Nat)

/-- What a capability-valued expression works out to, given what is held and what was minted.

    (held minted : Nat → Cap)      [the capabilities in scope, and the mint table]
    CExpr → Cap                    [takes an expression, gives a capability] -/
def CExpr.eval (held minted : Nat → Cap) : CExpr → Cap
  | .var i => held i
  | .mint i => minted i

/-- A capability is in scope when it is held or was minted.

    (held minted : Nat → Cap)                        [what is held, and the mint table]
    (c : Cap)                                        [a capability]
    (∃ i, held i = c) ∨ (∃ i, minted i = c)          [it is one of those two] -/
def InScope (held minted : Nat → Cap) (c : Cap) : Prop :=
  (∃ i, held i = c) ∨ (∃ i, minted i = c)

/-- The grammar is exactly its two constructors.

    (e : CExpr)                                      [any capability-valued expression]
    (∃ i, e = CExpr.var i) ∨ (∃ i, e = CExpr.mint i) [it is a read or a mint]

    So: the lemma that breaks first if a third constructor is ever added. -/
theorem cexpr_exhaustive (e : CExpr) :
    (∃ i, e = CExpr.var i) ∨ (∃ i, e = CExpr.mint i) := by
  cases e with
  | var i => exact Or.inl ⟨i, rfl⟩
  | mint i => exact Or.inr ⟨i, rfl⟩

/-- What a term can produce is exactly what is already in scope.

    (held minted : Nat → Cap)                        [what is held, and the mint table]
    (c : Cap)                                        [a capability]
    (∃ e : CExpr, e.eval held minted = c)            [some term produces it]
    ↔                                                [exactly when]
    InScope held minted c                            [it was held or minted]

    So: containment in both directions. Nothing a term produces is out of scope, which is the
    security property, and nothing in scope is beyond a term's reach, which says the grammar is
    exactly as expressive as intended rather than safe by being too weak. The predecessor
    proved the first direction only. -/
theorem reachable_iff_inScope (held minted : Nat → Cap) (c : Cap) :
    (∃ e : CExpr, e.eval held minted = c) ↔ InScope held minted c := by
  constructor
  · rintro ⟨e, he⟩
    subst he
    cases e with
    | var i => exact Or.inl ⟨i, rfl⟩
    | mint i => exact Or.inr ⟨i, rfl⟩
  · rintro (⟨i, hi⟩ | ⟨i, hi⟩)
    · exact ⟨CExpr.var i, hi⟩
    · exact ⟨CExpr.mint i, hi⟩

/-- Every capability a term produces is in scope.

    (held minted : Nat → Cap) (e : CExpr)            [what is held, the mint table, a term]
    InScope held minted (e.eval held minted)         [what it produces was already in scope]

    So: the hypothesis song's `Mandala.forge_reaches_only_scope` carries as `hUn`, discharged. -/
theorem eval_in_scope (held minted : Nat → Cap) (e : CExpr) :
    InScope held minted (e.eval held minted) :=
  (reachable_iff_inScope held minted _).mp ⟨e, rfl⟩

/-- Presenting a term as a capability reaches only granted regions, and only scoped values.

    {Region : Type} (held minted : Nat → Cap)        [what a region is, what is held and minted]
    (g : Grants Region) (e : CExpr) (r : Region)     [the grant table, a term, a region]
    (hlk : lookup g (e.eval held minted).index = some r)   [its index resolves to that region]
    InScope held minted (e.eval held minted)         [the value was already in scope]
    r ∈ g                                            [and the region was granted]
    ∧                                                [both together]

    So: no authentication hypothesis appears. The scope half does not come from the tag being
    right, it comes from there being no term that builds an out-of-scope capability. The lookup
    hypothesis is still needed for the region half. -/
theorem forge_reaches_only_scope {Region : Type} (held minted : Nat → Cap)
    (g : Grants Region) (e : CExpr) (r : Region)
    (hlk : lookup g (e.eval held minted).index = some r) :
    InScope held minted (e.eval held minted) ∧ r ∈ g :=
  ⟨eval_in_scope held minted e, lookup_mem g _ r hlk⟩

end Capability
end Apeiron
