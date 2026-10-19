import Gpu.Handoff

/-!
# The GPU handoff predicates' cost model (P84)

`Gpu.Handoff` carries the pure arithmetic of the Vulkan staging loop: the upload-window fit test,
the 256-byte alignment, the copy region, the scatter source, the binding count, the write chunk
and the barrier predicate. `src/gpu/runtime.rs` calls these functions where the staging loop
previously executed the expressions inline; each theorem below is a definitional equality between
the function and that inline expression, so the refactor adds no operation, no branch and no byte.

The staging loop's cost stays where it was: O(uploads) alignment/region steps and one O(1)
predicate per upload. The emitted command stream keeps its order (`Gpu.Handoff`'s
`stagedSubmission` and `uploadsFirst` theorems are the command-order contract; the refinement
theorem from the extracted Rust bodies to this model is open).

What is proved here: the predicates are their inline arithmetic, and the write chunk is bounded
by the cap and by the remaining bytes. What is not: a hardware timing; no theorem here produces a
cycle or a microsecond (obligation 11).
-/

namespace Perf.Handoff

open Gpu.Handoff

/-- The 256-byte alignment is the arithmetic the upload path executed inline. -/
theorem align256_eq (n : Nat) : align256 n = (n + 255) / 256 * 256 := rfl

/-- The fit test is the inline comparison. -/
theorem stageFit_eq (off len cap : Nat) : stageFit off len cap = decide (off + len ≤ cap) := rfl

/-- The next offset is the inline aligned add. -/
theorem stageNext_eq (off len : Nat) : stageNext off len = align256 (off + len) := rfl

/-- The copy region is the inline `(off, 0, len)` tuple. -/
theorem stageRegion_eq (off len : Nat) : stageRegion off len = (off, 0, len) := rfl

/-- The scatter source is the inline offset add. -/
theorem scatterSrc_eq (off s : Nat) : scatterSrc off s = off + s := rfl

/-- The bind check is the inline count comparison. -/
theorem bindsExactly_eq (n bindings : Nat) : bindsExactly n bindings = decide (n = bindings) := rfl

/-- The barrier predicate is the inline mode comparison. -/
theorem barEmits_eq (mode : Nat) : barEmits mode = (mode != 9) := rfl

/-- The write chunk is the inline `min (bytes - done) cap`, bounded by the cap. -/
theorem writeChunk_le_cap (done bytes cap : Nat) : writeChunk done bytes cap ≤ cap :=
  min_le_right _ _

/-- The write chunk is bounded by the remaining bytes. -/
theorem writeChunk_le_remaining (done bytes cap : Nat) : writeChunk done bytes cap ≤ bytes - done :=
  min_le_left _ _

end Perf.Handoff
