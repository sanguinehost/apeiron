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
-- release pin. This package compiles the corpus source off its own `lean-toolchain`; see README.
require «apeiron» from "../../../sanguine-apeiron/proof"

-- Must come last so Mathlib's transitive versions win over the corpus's stale manifest. Without
-- this the corpus drags in a 4.30.0-rc2-era Mathlib that does not build on Lean 4.31.0.
require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib ApeironWordBridge where
  roots := #[`WordRefinement, `AxiomAudit]
