import Lake
open Lake DSL

package apeironWordBridge where
  leanOptions := #[
    { name := `autoImplicit, value := false },
    { name := `warningAsError, value := true }
  ]

-- The extraction/denotation package next door.
require apeironWordProof from ".."

-- The canonical mathematical corpus at the exact revision refined by this bridge.
require «apeiron» from git "git@gitlab.tas.twn.network:lucas/sanguine.git" @
  "50c94192399add3b9f8e64bd14e36d8cba270705" / "proof"

-- Pin the same Mathlib revision as the corpus, last so this package's choice wins any transitive
-- resolution. The corpus is no longer on a 4.30.0-rc2-era Mathlib.
require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib ApeironWordBridge where
  roots := #[`WordRefinement, `TableRefinement, `TableRefinementGeneric,
    `TableRefinementBuild, `TableRefinementAdmission, `TableRefinementBuildGeneric,
    `TableRefinementReachablePrefixExactN, `TableRefinementAdmissionGeneric,
    `TableRefinementPhysical, `AxiomAudit,
    `Joint.TableInstance, `Joint.TableInstanceAudit, `Slang.Syntax, `Slang.Decode,
    `Slang.DecodeTest, `Slang.Word, `Slang.Machine, `Slang.MachineAudit]
