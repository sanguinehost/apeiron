/-
Axiom-cone check and carrier binding for the discrete flux theorems
(`docs/epics/magnetic-flux.md`).

The headline registration lives in `proof/corpus/AxiomAudit.lean`, next to the other bridge
headlines. That module transitively imports `Policy.Core`, which is dirty under another workstream,
so this module carries the same check inside the `Flux` library: the cone is verified at the
`lake build Flux` gate without waiting on a file owned by another task.

This module also binds the generic theorem to the corpus carrier. `RingCell R` is `abbrev` to
`R × R` (`Apeiron/Trichotomy/AnyRing.lean`), so the binding is a specialization, not a second proof.
-/
import Flux.Discrete
import Apeiron.AuditClosed
import Apeiron.Trichotomy.AnyRing

-- Registration and carrier-binding module: the same two options `AxiomAudit/Remaining.lean` sets.
-- The notation linter also reads the binder `j` as the split-complex unit.
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

#audit Flux.Discrete.sum_range_diff
#audit Flux.Discrete.grid_divergence_theorem
#audit Flux.Discrete.flux_zero_of_divergence_free

open Apeiron.Trichotomy.AnyRing

namespace Flux

/-- A divergence-free `RingCell`-valued face field has zero flux through the boundary of any block:
the lattice theorem at the corpus carrier. -/
theorem flux_zero_of_divergence_free_ringCell {R : Type*} [CommRing R]
    (Bx By : ℕ → ℕ → RingCell R) (m n : ℕ)
    (h : ∀ i j, Flux.Discrete.divergence Bx By i j = 0) :
    Flux.Discrete.boundaryFlux Bx By m n = 0 :=
  Flux.Discrete.flux_zero_of_divergence_free Bx By m n h

end Flux
