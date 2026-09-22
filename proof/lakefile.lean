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
--
-- The extraction needs model fixes the pinned release lacks: `Vec` carries a capacity field
-- (`capacity`/`try_reserve`/`try_reserve_exact`/`truncate`/`clear`), `Ord::max` takes its `Ord`
-- instance, and `Option::{as_ref,Try::branch,Try::from_residual}` / `Result::is_err` are modeled.
-- `proof/model-patches.sh` builds the patched backend at `target/aeneas-model/lean` from the
-- installed tree; the exact change is `proof/patches/aeneas-lean-model.patch`.
require aeneas from "../target/aeneas-model/lean"

@[default_target]
lean_lib ApeironWord where
  roots := #[`ApeironWord, `Refinement, `Denotation]
