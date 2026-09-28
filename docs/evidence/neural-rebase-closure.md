# Evidence: Stage-2 Neural Rebase Import Closure (APEIRON-NEURAL-CLOSURE)

- **Task:** `APEIRON-NEURAL-CLOSURE`
- **Grade:** `anchor`
- **Scope:** `docs/evidence/neural-rebase-closure.md` (only)
- **Source root for historical proof files:** `/home/socol/Workspace/sanguine/proof`
- **Reference spec:** [`docs/epics/neural-rebase.md`](../epics/neural-rebase.md)
- **Prior inventory:** [`docs/epics/neural-rebase.md`](../epics/neural-rebase.md)

```text
SCOPE:      docs/evidence/neural-rebase-closure.md; independent stage-2 closure recomputation
REGISTER:   planning / factual receipt (closure computation, digests, line counts, directory limits)
CORPUS:     canonical definitions Apeiron.Table.{HMap, get, insertWith, build} and
            Apeiron.Trichotomy.AnyRing.{RingCell, ringMul, ringNormForm}; historical modules at
            /home/socol/Workspace/sanguine/proof/
REFERENCE:  closure computed by breadth-first search over roots from /home/socol/Workspace/sanguine/proof
GATE:       cd /home/socol/Workspace/apeiron && git diff --no-index --check /dev/null docs/evidence/neural-rebase-closure.md
GRADE:      anchor
```

## Exact commands run

1. Python breadth-first closure computation script over the 6 Neural roots under `/home/socol/Workspace/sanguine/proof`.
2. `sha256sum` calculation and newline counting (`wc -l` equivalent) across all closure members.
3. Import inspection filtering `Mathlib*` and checking non-Mathlib references against the closure set.
4. Per-directory immediate file counting and maximum file length evaluation against AGENTS.md §4 limits.
5. Whitespace and path verification gate:
   ```sh
   cd /home/socol/Workspace/apeiron
   git diff --no-index --check /dev/null docs/evidence/neural-rebase-closure.md
   # Path existence check over every cited file
   ```

## Totals and comparison with specification

- **Module count:** 46 modules
- **Total lines:** 16,371 lines
- **Comparison with spec:**
  - Specification ([`docs/epics/neural-rebase.md`](../epics/neural-rebase.md) lines 20-23, line 38): states 46 modules, 16,371 lines.
  - Recomputed independent closure: exactly **46 modules** and **16,371 lines**.
  - Agreement status: **COMPLETE AGREEMENT** (0 module discrepancy, 0 line discrepancy).

## Unresolved imports outside closure

There are 108 unique import targets across the 46 modules in the closure:
- 42 imports resolve to internal modules within the closure set.
- 64 imports resolve to the external Mathlib distribution (excluded from closure traversal).
- Exactly **2** imports resolve outside both Mathlib and the local closure:

1. `Batteries`:
   - Imported by `/home/socol/Workspace/sanguine/proof/Substrate/SeedCore.lean:1` (`import Batteries`).
   - Resolved externally by the pinned Batteries dependency package (`proof/corpus/.lake/packages/batteries`).
2. `Std.Data.HashMap`:
   - Imported by `/home/socol/Workspace/sanguine/proof/Substrate/CollapseCore.lean:1` (`import Std.Data.HashMap`).
   - At the pinned Lean v4.31.0 toolchain, `Std.Data.HashMap` does not exist as an importable module name, while `Std.HashMap` and `Std.DHashMap` resolve from `Std.Data.DHashMap` (`/home/socol/.elan/toolchains/leanprover--lean4---v4.31.0/src/lean/Std/Data/DHashMap.lean`).
   - As documented in [`docs/epics/neural-rebase.md`](../epics/neural-rebase.md) §representation relation rule 2, `Substrate/CollapseCore.lean:1` requires adaptation from `import Std.Data.HashMap` to `import Std.Data.DHashMap`.

**Correction (executed compile).** The two unresolved-import entries above were classified by name
prefix, not by execution. An executed compile at the pinned identity finds: `import Std.Data.HashMap`
resolves (toolchain `Std/Data/HashMap.lean`; `cd proof/corpus && lake build Substrate.CollapseCore`
exit 0, 442 ms), and no import in the closure is unresolved. `lake build SubstrateVendored` fails at
three source-level sites — `Substrate/Algebra/QuantumField/MadelungCell.lean:214`,
`Substrate/Algebra/Translation/UniverseSyntax.lean:294`,
`Closures/Engineering/RadixDescent.lean:203/236/1157/1205` — which block 20 dependents. The digest and
line-count tables above are unaffected. Adaptation records are the plan rows
`APEIRON-NEURAL-W1-T0A/B/C`; the import-adaptation row is retired.

## Per-directory file counts and size limits

AGENTS.md §4 requires that no source file exceed 1,500 lines and no immediate directory exceed 12 immediate source files.

- **Maximum line count in closure:** 1,386 lines in `Closures/Engineering/RadixDescent.lean` (Limit: ≤1,500 lines → **PASS**).
- **Maximum immediate files per directory:** 10 files in `Substrate/Algebra/Bulk` (Limit: ≤12 files → **PASS**).

| Directory | Immediate files | File limit | File status | Max line file | Max lines | Line limit | Line status |
|---|---|---|---|---|---|---|---|
| `Closures/Engineering` | 2 | ≤12 | PASS | `RadixDescent.lean` | 1386 | ≤1500 | PASS |
| `Substrate` | 4 | ≤12 | PASS | `Seed.lean` | 402 | ≤1500 | PASS |
| `Substrate/Algebra/Bulk` | 10 | ≤12 | PASS | `BulkTower.lean` | 366 | ≤1500 | PASS |
| `Substrate/Algebra/Neural` | 7 | ≤12 | PASS | `TrainingCollapse.lean` | 830 | ≤1500 | PASS |
| `Substrate/Algebra/NullCell` | 3 | ≤12 | PASS | `Opportunity.lean` | 684 | ≤1500 | PASS |
| `Substrate/Algebra/QuantumField` | 3 | ≤12 | PASS | `QuantumClassicalIsomorphism.lean` | 386 | ≤1500 | PASS |
| `Substrate/Algebra/Security` | 2 | ≤12 | PASS | `HashTrinityCore.lean` | 533 | ≤1500 | PASS |
| `Substrate/Algebra/Translation` | 4 | ≤12 | PASS | `UniversalTranslation.lean` | 633 | ≤1500 | PASS |
| `Substrate/Algebra/Trinity` | 2 | ≤12 | PASS | `NonBeing.lean` | 281 | ≤1500 | PASS |
| `Substrate/Ski` | 5 | ≤12 | PASS | `PolyIotaTermAlgebra.lean` | 321 | ≤1500 | PASS |
| `Substrate/StationaryPhase` | 4 | ≤12 | PASS | `SpinorLoop.lean` | 1328 | ≤1500 | PASS |

All 11 directories pass the immediate file count limit (≤12) and the maximum line count limit (≤1,500 lines).

## Closure module inventory

The 46 modules traversed breadth-first starting from the 6 Neural roots:

| # | Module | Historical path (under `proof/`) | SHA-256 | Lines | Unresolved imports |
|---|---|---|---|---|---|
| 1 | `Substrate.Algebra.Neural.MirasCountLM` | `Substrate/Algebra/Neural/MirasCountLM.lean` | `9738ee17f7f7d130f69ab6c345743e7c2c06517d624a27d0cb4c7e6dd359577e` | 145 | none |
| 2 | `Substrate.Algebra.Neural.KneserNey` | `Substrate/Algebra/Neural/KneserNey.lean` | `4611c4071d735a8e3e6e6246da45257096b74f256fdd32e06dbb561ccdb6e768` | 172 | none |
| 3 | `Substrate.Algebra.Neural.CountLMFloor` | `Substrate/Algebra/Neural/CountLMFloor.lean` | `d82351a7d4dd167d79376abcc3fb5e43118ebe57c9a44fc0a1f2b3279d3c4fb9` | 537 | none |
| 4 | `Substrate.Algebra.Neural.AttentionSpinor` | `Substrate/Algebra/Neural/AttentionSpinor.lean` | `7cb47e7df62edc4bfb95ed4b20016563ee53537d92d388e39f8752c13b13935b` | 713 | none |
| 5 | `Substrate.Algebra.Neural.MLPrimitivesBridge` | `Substrate/Algebra/Neural/MLPrimitivesBridge.lean` | `32bba81f7b57dd38db1a23b475195bdeaaf8430c3a9402ba49f452ec012e50c0` | 452 | none |
| 6 | `Substrate.Algebra.Neural.TrainingCollapse` | `Substrate/Algebra/Neural/TrainingCollapse.lean` | `9eeb8663fcd765d7e73e1ce932cbd43ce979ccedb05095dc3ecb3247704d1c33` | 830 | none |
| 7 | `Substrate.Algebra.NullCell.NullCellCoding` | `Substrate/Algebra/NullCell/NullCellCoding.lean` | `3ddc17eb48a51cc272881a2be0d1fcee852a1d23cf1ea9aa60f05457c7888404` | 178 | none |
| 8 | `Substrate.Algebra.Neural.SparseAttentionBudget` | `Substrate/Algebra/Neural/SparseAttentionBudget.lean` | `a8510b998a17d3fcd25439035b2651e5c9403c9085f2748d7c29c05160a980eb` | 231 | none |
| 9 | `Substrate.StationaryPhase.SpinorLoop` | `Substrate/StationaryPhase/SpinorLoop.lean` | `c4bde98540dd61b2c18ed03a07ca5303cd4b9000b72fdb241adfe65836099ee7` | 1328 | none |
| 10 | `Substrate.Algebra.Security.HashTrinityCore` | `Substrate/Algebra/Security/HashTrinityCore.lean` | `9d14b4654ba6effba931cd400f456020e0273c635a0f8b2c0ad95b68183d9ff6` | 533 | none |
| 11 | `Substrate.Algebra.NullCell.Opportunity` | `Substrate/Algebra/NullCell/Opportunity.lean` | `3a96537ddea1571b9ac2ec3a7fd0e24f7623f2893315540732e47337adb39008` | 684 | none |
| 12 | `Substrate.Collapse` | `Substrate/Collapse.lean` | `18e1e90afc5ba37712336f7ad28f2b6e1b525a298f2ec0aa292d9ae163b2cd84` | 208 | none |
| 13 | `Substrate.Algebra.QuantumField.QuantumClassicalIsomorphism` | `Substrate/Algebra/QuantumField/QuantumClassicalIsomorphism.lean` | `b3a2ecddc8ad10861da3a17efaa030fabdec7a3350097fb6a34d98a4b46591c0` | 386 | none |
| 14 | `Substrate.Algebra.NullCell.NullCell` | `Substrate/Algebra/NullCell/NullCell.lean` | `7b64d26c870c0de685d69ea297599f9391e623a764ca1acb21c19e74a428010d` | 308 | none |
| 15 | `Substrate.Algebra.Bulk.BulkExhaustiveness` | `Substrate/Algebra/Bulk/BulkExhaustiveness.lean` | `85b911f8488e4446b26b4d9987c02e92a55560e742b9c0c2e12682593ae42e0c` | 300 | none |
| 16 | `Closures.Engineering.CoordinateReadout` | `Closures/Engineering/CoordinateReadout.lean` | `a958914d15f683ca2958684a359a1d4f5b172767803d7c69ae45fa52e04e7cf6` | 609 | none |
| 17 | `Substrate.StationaryPhase.AffineLoopCollapse` | `Substrate/StationaryPhase/AffineLoopCollapse.lean` | `bba8078a2431a80cd265dd0a9fddb31368097ee271a9d1cf22ae56f84d118c9f` | 154 | none |
| 18 | `Substrate.Ski.PolyIotaTermAlgebra` | `Substrate/Ski/PolyIotaTermAlgebra.lean` | `72ed9ef617226129ab65422f34e324f68c35eb7cc6af7e14a277a3c853709096` | 321 | none |
| 19 | `Substrate.Algebra.Trinity.NonBeing` | `Substrate/Algebra/Trinity/NonBeing.lean` | `ea20279feefa0224da64d793062ec816b7ec0ee2519e06858173141d482721ce` | 281 | none |
| 20 | `Substrate.StationaryPhase.BalancedSingularity` | `Substrate/StationaryPhase/BalancedSingularity.lean` | `fc63b519c762beaa7be4541b4fb2271f7350895f215ed17e71fdf6b8f5b819a5` | 110 | none |
| 21 | `Substrate.CollapseCore` | `Substrate/CollapseCore.lean` | `1a510e365f6374c571c215186f57cb2efcd31ed2f4530723c25ef21ff9e44cd3` | 106 | `Std.Data.HashMap` |
| 22 | `Substrate.Algebra.QuantumField.MadelungCell` | `Substrate/Algebra/QuantumField/MadelungCell.lean` | `2db1b90ea858edd8da5cf82358ea2a0b8b9363a59498c68703999933ebbaf9f9` | 246 | none |
| 23 | `Substrate.Algebra.QuantumField.WickRotation` | `Substrate/Algebra/QuantumField/WickRotation.lean` | `47e523d52647c679ed7fee4a6a843176fa818d27fe000c59693dfd8b26ab08a9` | 228 | none |
| 24 | `Substrate.Algebra.Security.EncryptionProjection` | `Substrate/Algebra/Security/EncryptionProjection.lean` | `7b4d09caaaf393a7988b8230e5436dbf10ee87d8754be4b640090f82556b53c2` | 283 | none |
| 25 | `Closures.Engineering.RadixDescent` | `Closures/Engineering/RadixDescent.lean` | `8cf237d66b671407857e245ef77782536a15087b8d276672d61eefa344a962bd` | 1386 | none |
| 26 | `Substrate.Ski.StateVectorMemoryModel` | `Substrate/Ski/StateVectorMemoryModel.lean` | `9520b1398e53cafca216ffb95d25124a2770595d1db238fa56f0465ef5895ab5` | 292 | none |
| 27 | `Substrate.Algebra.Bulk.BulkContains` | `Substrate/Algebra/Bulk/BulkContains.lean` | `db96594a4b2aaba5f0fda65e858c5c698920542c2116d225bd049ae4b86ca26c` | 312 | none |
| 28 | `Substrate.Algebra.Bulk.BulkTower` | `Substrate/Algebra/Bulk/BulkTower.lean` | `895a397dc60565991b7133aeb12746741a713815495db9dda923359620b415d7` | 366 | none |
| 29 | `Substrate.Ski.IotaTermAlgebra` | `Substrate/Ski/IotaTermAlgebra.lean` | `e32a10147596c732ad9a034bfbcf3f80c61b870cea3b3cb01616f045d6273719` | 183 | none |
| 30 | `Substrate.Ski.TermCore` | `Substrate/Ski/TermCore.lean` | `162994c5df5f99078db76bd4319c273946db559e7fd87f163b94c0642effd48f` | 123 | none |
| 31 | `Substrate.Algebra.Translation.UniverseSyntax` | `Substrate/Algebra/Translation/UniverseSyntax.lean` | `2a187804219ac504c33427a2dd778378b62faa4a2e85e8b9c4f78a22198a1e22` | 323 | none |
| 32 | `Substrate.Algebra.Trinity.BeingMonism` | `Substrate/Algebra/Trinity/BeingMonism.lean` | `9f47de3083db693b0a133173d7655c3b02e087619400037b8c68cba4de8b4860` | 146 | none |
| 33 | `Substrate.StationaryPhase.BiasInjectedRecovery` | `Substrate/StationaryPhase/BiasInjectedRecovery.lean` | `bb8691182eed507e546472ad280597269e88d50a499043f374f557c8260c45cc` | 237 | none |
| 34 | `Substrate.Algebra.Bulk.UniversalEquation` | `Substrate/Algebra/Bulk/UniversalEquation.lean` | `505ec7ff78767939bf9b9d7452dba069385662a110ffc844d4160725974a73e7` | 274 | none |
| 35 | `Substrate.Ski.StateVectorProjection` | `Substrate/Ski/StateVectorProjection.lean` | `d7c7ba9fd3a5f79116e19f1ec73c67ba4f22981a1d70858c837ca222c2ca302b` | 109 | none |
| 36 | `Substrate.Algebra.Bulk.BulkInevitability` | `Substrate/Algebra/Bulk/BulkInevitability.lean` | `e52fc5164d7d2c79c99e535c5b67eff3528273b9b26996953019bbd2a235fa6a` | 237 | none |
| 37 | `Substrate.Algebra.Translation.UniversalTranslation` | `Substrate/Algebra/Translation/UniversalTranslation.lean` | `de4a288aa18a7e9d12972d4502f76ad6ae75ddec51e735d0445950174f78b100` | 633 | none |
| 38 | `Substrate.Algebra.Bulk.FreeBulkAlgebra` | `Substrate/Algebra/Bulk/FreeBulkAlgebra.lean` | `782aa7fa12253244ff224b78c1f04547fc4342701d937ef81d86ce50f48b28b2` | 186 | none |
| 39 | `Substrate.Algebra.Bulk.AbstractBulkDifferential` | `Substrate/Algebra/Bulk/AbstractBulkDifferential.lean` | `f0ac9f27342db088feeba1864ad6735728a2c47b8043f393ec8cfe48f5e8f23d` | 181 | none |
| 40 | `Substrate.Algebra.Bulk.UniversalEncoding` | `Substrate/Algebra/Bulk/UniversalEncoding.lean` | `17854cb949063e866051886d83bf66895fca65a8d45c92a7d8fbead4ca0e0b90` | 135 | none |
| 41 | `Substrate.Seed` | `Substrate/Seed.lean` | `8bdd58547afc9ce1247f0846b23e86b6b32d55bd765ce98c7dcdb6b284d921c2` | 402 | none |
| 42 | `Substrate.Algebra.Translation.UniversalApplicativeLift` | `Substrate/Algebra/Translation/UniversalApplicativeLift.lean` | `cebc18922b9bd3990f0f0e51feaf89cee97aad9949130638e4bc6e4f17f7afdf` | 336 | none |
| 43 | `Substrate.Algebra.Translation.UniverseSyntaxSerialisation` | `Substrate/Algebra/Translation/UniverseSyntaxSerialisation.lean` | `2f1e95e05e92f21ebf9b43f9d064560616c3fc3d68c1386efdc73c98004b0ebb` | 543 | none |
| 44 | `Substrate.SeedCore` | `Substrate/SeedCore.lean` | `587151458d90cfa599d2dfcd05c71a5b76608f475d490e99f9178c637b508eb4` | 342 | `Batteries` |
| 45 | `Substrate.Algebra.Bulk.DiscreteApplicative` | `Substrate/Algebra/Bulk/DiscreteApplicative.lean` | `7d7fcb600009cc23eebb44c0847c16e2322b6d906f3d4e1f7931bf9997900a50` | 138 | none |
| 46 | `Substrate.Algebra.Bulk.DiscreteArithmetic` | `Substrate/Algebra/Bulk/DiscreteArithmetic.lean` | `43287bb4986d37eb540a46cb3eb1ab1fb31d87ff5158da21b546e3373a64ee7f` | 144 | none |

## Reproducibility

The closure was computed using the following standalone script executed with Python 3:

```python
import os
import hashlib
from collections import deque, defaultdict

proof_root = "/home/socol/Workspace/sanguine/proof"

# 1. Index all .lean files excluding .lake
rel_path_to_mod = {}
mod_to_rel_path = {}
for root, dirs, files in sorted(os.walk(proof_root)):
    if ".lake" in root.split(os.sep):
        continue
    for file in sorted(files):
        if file.endswith(".lean"):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, proof_root)
            mod_name = ".".join(rel_path[:-5].split(os.sep))
            rel_path_to_mod[rel_path] = mod_name
            mod_to_rel_path[mod_name] = rel_path

def parse_imports(filepath):
    imports = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("import "):
                parts = line_str.split()
                if len(parts) >= 2:
                    imports.append(parts[1])
    return imports

roots = [
    "Substrate.Algebra.Neural.MirasCountLM",
    "Substrate.Algebra.Neural.KneserNey",
    "Substrate.Algebra.Neural.CountLMFloor",
    "Substrate.Algebra.Neural.AttentionSpinor",
    "Substrate.Algebra.Neural.MLPrimitivesBridge",
    "Substrate.Algebra.Neural.TrainingCollapse",
]

# 2. Breadth-first traversal
queue = deque(roots)
closure_modules = []
seen = set(roots)

while queue:
    curr = queue.popleft()
    closure_modules.append(curr)
    rel_p = mod_to_rel_path[curr]
    full_p = os.path.join(proof_root, rel_p)
    for imp in parse_imports(full_p):
        if imp.startswith("Mathlib"):
            continue
        if imp in mod_to_rel_path and imp not in seen:
            seen.add(imp)
            queue.append(imp)

closure_set = set(closure_modules)
total_lines = 0
per_dir_files = defaultdict(list)
max_lines = 0
max_file = None

# 3. Digests, line counts, unresolved imports, and limits
for mod in closure_modules:
    rel_p = mod_to_rel_path[mod]
    full_p = os.path.join(proof_root, rel_p)
    with open(full_p, "rb") as f:
        content = f.read()
    digest = hashlib.sha256(content).hexdigest()
    lines = content.count(b"\n")
    imps = parse_imports(full_p)
    unresolved = [imp for imp in imps if not imp.startswith("Mathlib") and imp not in closure_set]
    dir_name = os.path.dirname(rel_p)
    per_dir_files[dir_name].append((rel_p, lines))
    if lines > max_lines:
        max_lines = lines
        max_file = rel_p
    total_lines += lines

print(f"Total modules: {len(closure_modules)}")
print(f"Total lines: {total_lines}")
print(f"Max lines: {max_lines} in {max_file}")
```

## Gate execution and verification

The receipt is validated by running the whitespace check and the path existence check:

```sh
cd /home/socol/Workspace/apeiron
git diff --no-index --check /dev/null docs/evidence/neural-rebase-closure.md
```

Gate output logged to `/tmp/opencode/neural-closure-gate.log`:
- Whitespace diagnostic check: clean exit 1 (zero whitespace errors).
- Path existence check (`test -e` loop over all cited paths): 63 paths verified with zero `MISSING`.
