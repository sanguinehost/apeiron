import Lake
open Lake DSL

package apeironWordProof where
  leanOptions := #[
    { name := `autoImplicit, value := false },
    { name := `warningAsError, value := true }
  ]

-- Aeneas's Lean backend is deliberately not vendored. Install the pinned archive under
-- `target/aeneas/` (see README), or point `AENEAS_HOME` at an existing installation and let
-- `regen.sh` wire it in. This is the same relative layout Penglai uses.
require aeneas from "../target/aeneas/backends/lean"

@[default_target]
lean_lib ApeironWord where
  roots := #[`ApeironWord, `Refinement, `Denotation]
