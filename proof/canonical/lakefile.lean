import Lake
open Lake DSL

package «apeiron» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`warningAsError, true⟩
  ]

require "leanprover-community" / "mathlib" @ git "v4.31.0"

/-- The library. `Apeiron.lean` is the single root; `Apeiron.*` is glob'd, so a new file
    under `Apeiron/` is in the build the moment it exists: no root list to maintain. -/
@[default_target]
lean_lib «Apeiron» where
  srcDir := "."
  globs := #[`Apeiron, `Apeiron.*]
