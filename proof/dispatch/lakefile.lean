import Lake
open Lake DSL
package dispatchProof where
  leanOptions := #[⟨`autoImplicit, false⟩, ⟨`warningAsError, true⟩]
require aeneas from "../../target/aeneas-model/lean"
@[default_target]
lean_lib Dispatch where
  roots := #[`Dispatch.Extracted, `Dispatch.Spec, `Dispatch.Refinement, `Dispatch.Audit]
