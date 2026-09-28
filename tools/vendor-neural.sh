#!/usr/bin/env bash
# Vendor the stage-2 neural rebase import closure from the historical Sanguine
# proof tree into proof/corpus/.
#
# The source list is the 46-module closure recorded in
# docs/evidence/neural-rebase-closure.md (APEIRON-NEURAL-CLOSURE); it is
# hardcoded so the copy is deterministic and does not depend on a runtime
# search of the source tree. Copies are byte-identical; the manifest
# proof/corpus/Substrate/vendor.sha256 records the expected digests.
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/home/socol/Workspace/sanguine/proof}"
DST_ROOT="${DST_ROOT:-/home/socol/Workspace/apeiron/proof/corpus}"

files=(
  "Substrate/Algebra/Neural/MirasCountLM.lean"
  "Substrate/Algebra/Neural/KneserNey.lean"
  "Substrate/Algebra/Neural/CountLMFloor.lean"
  "Substrate/Algebra/Neural/AttentionSpinor.lean"
  "Substrate/Algebra/Neural/MLPrimitivesBridge.lean"
  "Substrate/Algebra/Neural/TrainingCollapse.lean"
  "Substrate/Algebra/Neural/FrontierLadder.lean"
  "Substrate/Algebra/NullCell/NullCellCoding.lean"
  "Substrate/Algebra/NullCell/SpinorFold.lean"
  "Substrate/Algebra/Neural/SparseAttentionBudget.lean"
  "Substrate/StationaryPhase/SpinorLoop.lean"
  "Substrate/Algebra/Security/HashTrinityCore.lean"
  "Substrate/Algebra/NullCell/Opportunity.lean"
  "Substrate/Collapse.lean"
  "Substrate/Algebra/QuantumField/QuantumClassicalIsomorphism.lean"
  "Substrate/Algebra/NullCell/NullCell.lean"
  "Substrate/Algebra/Bulk/BulkExhaustiveness.lean"
  "Closures/Engineering/CoordinateReadout.lean"
  "Substrate/StationaryPhase/AffineLoopCollapse.lean"
  "Substrate/Ski/PolyIotaTermAlgebra.lean"
  "Substrate/Algebra/Trinity/NonBeing.lean"
  "Substrate/StationaryPhase/BalancedSingularity.lean"
  "Substrate/CollapseCore.lean"
  "Substrate/Algebra/QuantumField/MadelungCell.lean"
  "Substrate/Algebra/QuantumField/WickRotation.lean"
  "Substrate/Algebra/Security/EncryptionProjection.lean"
  "Closures/Engineering/RadixDescent.lean"
  "Substrate/Ski/StateVectorMemoryModel.lean"
  "Substrate/Algebra/Bulk/BulkContains.lean"
  "Substrate/Algebra/Bulk/BulkTower.lean"
  "Substrate/Ski/IotaTermAlgebra.lean"
  "Substrate/Ski/TermCore.lean"
  "Substrate/Algebra/Translation/UniverseSyntax.lean"
  "Substrate/Algebra/Trinity/BeingMonism.lean"
  "Substrate/StationaryPhase/BiasInjectedRecovery.lean"
  "Substrate/Algebra/Bulk/UniversalEquation.lean"
  "Substrate/Ski/StateVectorProjection.lean"
  "Substrate/Algebra/Bulk/BulkInevitability.lean"
  "Substrate/Algebra/Translation/UniversalTranslation.lean"
  "Substrate/Algebra/Bulk/FreeBulkAlgebra.lean"
  "Substrate/Algebra/Bulk/AbstractBulkDifferential.lean"
  "Substrate/Algebra/Bulk/UniversalEncoding.lean"
  "Substrate/Seed.lean"
  "Substrate/Algebra/Translation/UniversalApplicativeLift.lean"
  "Substrate/Algebra/Translation/UniverseSyntaxSerialisation.lean"
  "Substrate/SeedCore.lean"
  "Substrate/Algebra/Bulk/DiscreteApplicative.lean"
  "Substrate/Algebra/Bulk/DiscreteArithmetic.lean"
)

for rel in "${files[@]}"; do
  src="$SRC_ROOT/$rel"
  dst="$DST_ROOT/$rel"
  if [[ ! -f "$src" ]]; then
    printf 'vendor-neural: missing source %s\n' "$src" >&2
    exit 1
  fi
  install -D -m 0644 "$src" "$dst"
done
printf 'vendor-neural: copied %d modules\n' "${#files[@]}"

# A re-run must reproduce the adapted tree: verify the fresh copies against the historical
# manifest, apply the recorded adaptations, then verify the final tree against the adapted
# manifest. Patch labels are repo-root-relative (`a/proof/corpus/<rel>`), applied with git apply
# from the repository root.
REPO_ROOT="$(cd "$DST_ROOT/../.." && pwd)"
PRISTINE_MANIFEST="proof/corpus/Substrate/vendor.sha256"
ADAPTED_MANIFEST="proof/corpus/Substrate/adapted.sha256"

(cd "$REPO_ROOT" && sha256sum --quiet -c "$PRISTINE_MANIFEST")
printf 'vendor-neural: pristine manifest verified (%s)\n' "$PRISTINE_MANIFEST"

shopt -s nullglob
for patch in "$REPO_ROOT"/tools/neural-adapt/*.patch; do
  git -C "$REPO_ROOT" apply --whitespace=nowarn "$patch"
  printf 'vendor-neural: applied %s\n' "${patch#"$REPO_ROOT"/}"
done
shopt -u nullglob

if [[ -f "$REPO_ROOT/$ADAPTED_MANIFEST" ]]; then
  (cd "$REPO_ROOT" && sha256sum --quiet -c "$ADAPTED_MANIFEST")
  printf 'vendor-neural: adapted manifest verified (%s)\n' "$ADAPTED_MANIFEST"
fi
