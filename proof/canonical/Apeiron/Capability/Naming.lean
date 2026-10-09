import Apeiron.Linter.Notation

/-!
# The naming map reaches ungranted interior, and the extent fold closes it

@name Naming
@grade proved

`Capability/Basic.lean` is about the capabilities a program can *build*; `Capability/Extent.lean`
is about what a per-cell table can *authenticate*. This file is about the third surface between
them: the map from a raw key to the cell it names, `memory[k]`.

The emitted map is `k ↦ HEAP_BASE + (k and 2^W-1) * 8`. The mask `k and 2^W-1` is `k % 2^W`, so
the image of the coordinate half is the whole window `[0, 2^W)` -- every residue is hit, by the key
equal to it. That image is fixed by the mask width `W` alone. It does not consult the grant table,
so it cannot exclude a coordinate the grants omit. When the runtime keeps its own control state in a
cell inside the window that no program grant covers, that cell is named by the key equal to its
coordinate. `nameMap_surjective` and `ungranted_is_nameable` state exactly this: masking contains
the access inside the window and destroys the information that would tell a granted coordinate
from an ungranted one, which is containment, not authentication -- the same shape `Extent.lean`
proves for the size table, here for the address itself.

The fix is a second collapse composed after the first: read the per-cell extent at the masked
coordinate and, when it is zero (ungranted), rewrite the *whole* address to a fixed sink cell in the
unaddressable band. `foldMap` is that composite, and `ungranted_folds_to_sink` says no ungranted
coordinate names an arena cell any more -- it folds to the sink, exactly as the mask already folds
the coordinate. The sink sits below the window, so no key names it either.

## What this does not settle

That a particular compiler emits the bare masked map rather than the folded one. That is read from
its source, not established here. This file establishes only that the bare map reaches ungranted
interior wherever the grants do not fill the window, and that the extent fold removes every such
reach.
-/

namespace Apeiron
namespace Capability

/-- The coordinate a key names: the key folded into the window by the mask.

    (W k : Nat)          [the mask width, and a raw key]
    Nat                  [the coordinate in `[0, 2^W)` that the key names]

    The mask `k and 2^W-1` is `k % 2^W`; this is that fold, the whole content of the address map's
    coordinate half. -/
def nameMap (W k : Nat) : Nat := k % 2 ^ W

/-- The fold lands in the window.

    (W k : Nat)                  [the mask width, and a key]
    nameMap W k < 2 ^ W          [the named coordinate is inside the window] -/
theorem nameMap_lt (W k : Nat) : nameMap W k < 2 ^ W :=
  Nat.mod_lt k Nat.one_le_two_pow

/-- Every coordinate in the window is named, by the key equal to it.

    (W c : Nat)                          [the mask width, and a target coordinate]
    (hc : c < 2 ^ W)                     [the coordinate sits inside the window]
    nameMap W c = c                      [the key `c` names coordinate `c`]

    So: the coordinate half is surjective onto `[0, 2^W)`. Its image is the whole window, fixed by
    `W` alone. -/
theorem nameMap_surjective (W c : Nat) (hc : c < 2 ^ W) : nameMap W c = c :=
  Nat.mod_eq_of_lt hc

/-- A control-state cell inside the window that no grant covers is named all the same.

    (W ctl : Nat)                        [the mask width, and a control-state coordinate]
    (granted : Nat → Bool)               [which coordinates a program grant covers]
    (hin : ctl < 2 ^ W)                  [the control cell sits inside the window]
    (hun : granted ctl = false)          [and is covered by no grant]
    nameMap W ctl = ctl                             [the key `ctl` names it]
    ∧ granted (nameMap W ctl) = false               [and the named cell is ungranted]

    So: the map hands back a live ungranted cell -- the ring band, the saved entry rsp -- because
    the image is the window and the window ignores the grants. Containment, not authentication. -/
theorem ungranted_is_nameable (W ctl : Nat) (granted : Nat → Bool)
    (hin : ctl < 2 ^ W) (hun : granted ctl = false) :
    nameMap W ctl = ctl ∧ granted (nameMap W ctl) = false := by
  have h : nameMap W ctl = ctl := nameMap_surjective W ctl hin
  refine ⟨h, ?_⟩
  rw [h]; exact hun

/-- The folded map: the masked coordinate when its per-cell extent is positive, the sink otherwise.

    (W sink : Nat) (extent : Nat → Nat) (k : Nat)   [the width, a sink cell, the extents, a key]
    Nat                                             [the coordinate the folded map names]

    `extent` reads 0 at an ungranted cell (`Extent.lean`'s table, stamped from the grants). When it
    is 0 the whole address is rewritten to `sink`; otherwise the ordinary masked coordinate stands.
    A second collapse, composed after the mask. -/
def foldMap (W sink : Nat) (extent : Nat → Nat) (k : Nat) : Nat :=
  if extent (nameMap W k) = 0 then sink else nameMap W k

/-- A granted cell folds to exactly the masked coordinate the bare map named.

    (W sink : Nat) (extent : Nat → Nat) (k : Nat)   [the width, the sink, the extents, a key]
    (hpos : 0 < extent (nameMap W k))               [the named cell was granted a positive extent]
    foldMap W sink extent k = nameMap W k           [so the fold is the identity there]

    So: the fold changes nothing on a granted access -- the collapse only fires where the extent is
    zero. -/
theorem foldMap_granted (W sink : Nat) (extent : Nat → Nat) (k : Nat)
    (hpos : 0 < extent (nameMap W k)) :
    foldMap W sink extent k = nameMap W k := by
  have h : extent (nameMap W k) ≠ 0 := Nat.pos_iff_ne_zero.mp hpos
  simp [foldMap, h]

/-- No ungranted coordinate names an arena cell: it folds to the sink.

    (W sink : Nat) (extent : Nat → Nat) (k : Nat)   [the width, the sink, the extents, a key]
    (hun : extent (nameMap W k) = 0)                [the named cell is ungranted, extent 0]
    foldMap W sink extent k = sink                  [the whole address is the sink]

    So: the reach `ungranted_is_nameable` exhibited is gone. Every key whose masked coordinate is
    ungranted names the one sink cell, which sits below the window (`sink < 2^W` is false by
    construction of the band) and so is itself unnameable. The hole closes by a collapse, not a
    check. -/
theorem ungranted_folds_to_sink (W sink : Nat) (extent : Nat → Nat) (k : Nat)
    (hun : extent (nameMap W k) = 0) :
    foldMap W sink extent k = sink := by
  simp [foldMap, hun]

end Capability
end Apeiron
