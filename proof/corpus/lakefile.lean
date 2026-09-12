import Lake
open Lake DSL

package apeironWordBridge where
  leanOptions := #[
    { name := `autoImplicit, value := false },
    { name := `warningAsError, value := true }
  ]

-- The extraction/denotation package next door.
require apeironWordProof from ".."

-- The canonical mathematical corpus. The sibling path is a development convenience, not a
-- release pin. The corpus at 16112030 pins Lean 4.31.0 and Mathlib v4.31.0 and carries the
-- matching source adjustment in Apeiron/Lattice/EvenUnimodular.lean; see README.
require «apeiron» from "../../../sanguine-apeiron/proof"

-- Pin the same Mathlib revision as the corpus, last so this package's choice wins any transitive
-- resolution. The corpus is no longer on a 4.30.0-rc2-era Mathlib.
require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib ApeironWordBridge where
  roots := #[`WordRefinement, `AxiomAudit]
