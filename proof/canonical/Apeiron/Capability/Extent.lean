import Apeiron.Linter.Notation

/-!
# The extent table cannot authenticate a coordinate

@name Extent
@grade proved

`Capability/Basic.lean` locates the provenance guarantee in the grammar: no term builds a
capability out of arithmetic, so every capability presented at a boundary was one that was
handed over. `forge_reaches_only_scope` there says the deref itself decides nothing -- the
region half rests on a table lookup, not on the value the program holds.

This file is about that table, and about why it is containment and not authentication.

A real compiler answers "what extent was granted at this coordinate?" with a finite table. The
coordinate is `n` bits wide; the table has `2^m` slots with `m < n`, so the coordinate is
folded by its low bits -- reduced modulo `P = 2^m` -- before the table is read. That fold is a
collapse: it keeps every access inside the table, and it destroys the information that would
tell two coordinates in the same residue class apart.

The consequence is stated as `ungranted_reads_granted_extent`. Take a coordinate that was
granted a positive extent inside the first period. The coordinate one period higher was never
granted anything -- it is past the whole arena -- yet the lookup returns the granted coordinate's
positive extent, because the two share a residue. So a clamp built on this lookup bounds a
transfer at a coordinate it never authorised. The table decided the residue class, which is a
containment, and mistook it for the coordinate, which would have been authentication.

Nothing here mentions `Cell`, `mul` or σ. It is arithmetic about a masked index, and
that is the whole point: masking is the wrong tool for provenance, and the reason is elementary.

## What this does not settle

That a particular compiler sizes its table below its coordinate space. That is a fact about
that compiler, read from its source, and not established here. This file establishes only that
where it holds, the lookup cannot recover provenance.
-/

namespace Apeiron
namespace Capability

/-- The lookup the emitted clamp performs.

    (P : Nat)                       [how many slots the table has, its period]
    (size : Nat → Nat)              [the extent recorded there, 0 where nothing was granted]
    (offset : Nat)                  [the coordinate, as an offset from the arena base]
    Nat                             [gives the extent the table returns]

    The offset is folded by `offset % P` before the table is read, because the table has only
    `P` slots. That fold is the whole mechanism this file is about. -/
def extentLookup (P : Nat) (size : Nat → Nat) (offset : Nat) : Nat := size (offset % P)

/-- The lookup cannot tell a coordinate from the one a period higher.

    (P : Nat) (size : Nat → Nat) (offset : Nat)   [the period, the extents, a coordinate]
    extentLookup P size (offset + P) = extentLookup P size offset [same at the aliasing coordinate]

    So: one line, the residue is all the table sees. -/
theorem lookup_period (P : Nat) (size : Nat → Nat) (offset : Nat) :
    extentLookup P size (offset + P) = extentLookup P size offset := by
  simp [extentLookup, Nat.add_mod_right]

/-- A granted coordinate, read below its period, reads exactly its recorded extent.

    (P : Nat) (size : Nat → Nat) (offset : Nat)   [the period, the extents, a coordinate]
    (hin : offset < P)                            [the coordinate sits below the period]
    extentLookup P size offset = size offset                  [so the fold is the identity here]

    So: the fold changes nothing inside the first period, which is where the arena is stamped. -/
theorem lookup_below_period (P : Nat) (size : Nat → Nat) (offset : Nat) (hin : offset < P) :
    extentLookup P size offset = size offset := by
  simp [extentLookup, Nat.mod_eq_of_lt hin]

/-- The lookup returns a granted, positive extent for a coordinate that was never granted.

    (P span : Nat) (size : Nat → Nat) (offset : Nat)   [period, arena width, extents, coordinate]
    (hP : 0 < P)                                       [the table has at least one slot]
    (hspan : span ≤ P)                              [the whole granted arena fits inside one period]
    (hin : offset < span)                              [the coordinate is inside the granted arena]
    (hpos : 0 < size offset)                           [and was granted a positive extent]
    span ≤ offset + P                [the aliasing coordinate is past the whole arena, so ungranted]
    ∧ offset + P ≠ offset                              [it is a different coordinate]
    ∧ extentLookup P size (offset + P) = extentLookup P size offset   [reads the same as at offset]
    ∧ 0 < extentLookup P size (offset + P)                   [and that read is positive]

    So: containment, not authentication. The clamp bounds a transfer at `offset + P`, a
    coordinate outside every grant, to the extent granted at `offset`. This is the arithmetic
    behind a per-cell extent table sized below its coordinate space: the fold decides the
    residue class and the clamp treats it as the coordinate. -/
theorem ungranted_reads_granted_extent
    (P span : Nat) (size : Nat → Nat) (offset : Nat)
    (hP : 0 < P) (hspan : span ≤ P) (hin : offset < span) (hpos : 0 < size offset) :
    span ≤ offset + P
      ∧ offset + P ≠ offset
      ∧ extentLookup P size (offset + P) = extentLookup P size offset
      ∧ 0 < extentLookup P size (offset + P) := by
  refine ⟨?_, ?_, lookup_period P size offset, ?_⟩
  · exact Nat.le_trans hspan (Nat.le_add_left P offset)
  · omega
  · have hlt : offset < P := Nat.lt_of_lt_of_le hin hspan
    rw [lookup_period P size offset, lookup_below_period P size offset hlt]
    exact hpos

/-! ## The fix: a bounds-checked index does not alias

The fold aliases because it wraps. The fix is not a bigger table -- the table already holds `span`
slots -- it is to replace the wrap with a bounds check: index directly when the coordinate is inside
the stamped window `[0, span)`, and read 0 (ungranted) otherwise. This is well-defined exactly
because the compiler sizes its table at or above its grantable window, so no in-window coordinate is
ever thrown away by the check. -/

/-- The lookup the fixed clamp performs: the extent inside the window, 0 outside it.

    (span : Nat) (size : Nat → Nat) (offset : Nat)   [the window width, the extents, a coordinate]
    Nat                                           [the extent, 0 for any coordinate past the window]

    No fold. A coordinate at or past `span` was granted nothing and reads 0 directly, instead of
    borrowing a residue-mate's extent. -/
def extentChecked (span : Nat) (size : Nat → Nat) (offset : Nat) : Nat :=
  if offset < span then size offset else 0

/-- Inside the window the checked lookup is the table, unchanged.

    (span : Nat) (size : Nat → Nat) (offset : Nat)   [the window, the extents, a coordinate]
    (hin : offset < span)                            [the coordinate is inside the window]
    extentChecked span size offset = size offset     [so the check is the identity here]

    So: every granted access reads exactly what was stamped -- a granted read costs nothing. -/
theorem checked_in_window (span : Nat) (size : Nat → Nat) (offset : Nat) (hin : offset < span) :
    extentChecked span size offset = size offset := by
  simp [extentChecked, hin]

/-- A coordinate past the window reads 0, so no ungranted coordinate borrows a granted extent.

    (span : Nat) (size : Nat → Nat) (offset : Nat)   [the window, the extents, a coordinate]
    (hout : span ≤ offset)                           [the coordinate is at or past the window]
    extentChecked span size offset = 0               [the clamp bounds it to nothing]

    So: the exact reach `ungranted_reads_granted_extent` exhibited is gone. Where the fold
    returned a granted, positive extent for `offset + P`, the checked lookup returns 0, because
    `span ≤ offset` fails the bounds test. The clamp now bounds an ungranted transfer to zero --
    safe, not silent. -/
theorem checked_ungranted_reads_zero (span : Nat) (size : Nat → Nat) (offset : Nat)
    (hout : span ≤ offset) :
    extentChecked span size offset = 0 := by
  simp [extentChecked, Nat.not_lt.mpr hout]

/-- Each in-window coordinate reads its OWN stamped extent, so no two are confused.

    (span : Nat) (size : Nat → Nat) (a b : Nat)      [the window, the extents, two coordinates]
    (ha : a < span) (hb : b < span)                  [both inside the window]
    extentChecked span size a = size a               [`a` reads the extent stamped at `a`]
    ∧ extentChecked span size b = size b             [`b` reads the extent stamped at `b`]

    So: the degree-16 covering is degree 1 inside the window. Where the fold made `a` and `a + P`
    read one residue, the checked lookup reads each coordinate's own slot; distinct coordinates are
    never routed to a shared extent. The aliasing was the fold's, not the table's. -/
theorem checked_no_alias (span : Nat) (size : Nat → Nat) (a b : Nat)
    (ha : a < span) (hb : b < span) :
    extentChecked span size a = size a ∧ extentChecked span size b = size b :=
  ⟨checked_in_window span size a ha, checked_in_window span size b hb⟩

end Capability
end Apeiron
