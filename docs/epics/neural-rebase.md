# Neural rebase (verified-AI sequence, stage 2)

Merged epic record: `intent/neural-rebase.md`, `spec/neural-rebase.md`, `plan/neural-rebase.md`, `plan/neural-rebase-inventory.md` (one file per epic; see `docs/README.md`).

## Intent — Neural rebase (verified-AI sequence, stage 2)

- **Owner:** Apeiron lead
- **Source revision:** `74c037b`; working tree dirty (in-flight `Growable/Read.lean`,
  `Growable/RelocationWF.lean` and `proof/corpus/lakefile.lean`; uncommitted `Growable/Classifier.lean`,
  `AxiomAudit.lean`, the docs in this change set)
- **Status:** draft
- **Spec:** [`docs/epics/neural-rebase.md`](neural-rebase.md) (draft)
- **Plan:** [`docs/epics/neural-rebase.md`](neural-rebase.md) (draft)

```text
SCOPE: docs/epics/neural-rebase.md only; planning, no source, theorem or gate change
REGISTER: refinement planning for the AI-track outcome
          (docs/direction.md#programme-outcomes, "Active test-time learning and lower-cost useful
          AI"); excludes execution, grade changes, pin changes and performance work
CORPUS: canonical definitions Apeiron.Table.{HMap, get, insertWith, build} and
        Apeiron.Trichotomy.AnyRing.{RingCell, ringMul, ringNormForm} at the pinned corpus
        (../sanguine-apeiron/proof, corpus Git 3d66ea3aa2fe1de5997b6b757247454a2f771358 per the
        bridge manifest); historical statements in ../sanguine/proof/Substrate/Algebra/Neural/ and
        ../sanguine/proof/Substrate/Algebra/NullCell/SpinorFold.lean; paper1/paper2 bundles;
        ../spinor-ttl ports
REFERENCE: docs/direction.md outcomes row and execution order; docs/roadmap.md
           "Verified-AI sequence and prior-art index"; the module files listed above, read in this
           change set; no theorem is claimed here
GATE: git diff --check on this file; no Lean gate is run by this planning document
GRADE: anchor/planning; historical grades retained, not fresh acceptance
```

### Problem

The AI-track outcome's formal records live in sibling checkouts at the older toolchain
(`../sanguine` pins `leanprover/lean4:v4.30.0-rc2`; this package pins `v4.31.0`), and no rebased
statement links a streaming-update or attention mechanism to the canonical
`Apeiron.Table`/`Apeiron.Trichotomy` objects consumed here. The executing reference is the
`../spinor-ttl` port suite: measured, not extracted. `docs/direction.md` records the state:
"Existing work is not new acceptance." Stage 3 (end-to-end extraction) has no admissible input until
a stage-2 artifact re-checks the mechanism statements at the current identity and names their link
to the canonical object.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Cite paper1/paper2/spinor-ttl and start stage 3 | A citation is a pointer, not evidence (AGENTS.md §5); an extracted primitive would refine no current theorem. |
| Copy the historical modules in and relabel their grades | A relabel is not a re-check; the modules pin the older Mathlib, and mixing toolchains in the shared corpus cache is a recorded failure ([receipt](../evidence/lean-toolchain-mixing-incident.md)). |
| Re-derive the mechanisms from scratch | Discards the reviewed statements that serve as the specification and duplicates work the rebase can re-bind. |
| Treat the spinor-ttl measurements as the mechanism evidence | Measurements are instrument-bound; they establish observed effects, not statements over the canonical definitions. |
| Move the canonical corpus now | Migration is its own open outcome row ("Retained Lean + reusable Rust source convergence"); the pin is unchanged and this epic does not preempt it. |

### Completion condition

The spec's rebase table has one row per admitted module: the row's focused module gate exits 0 under
the pinned toolchain, every headline is registered in `AxiomAudit` inside
`{propext, Classical.choice, Quot.sound}`, and the row names its historical source file with its
sha256 plus either its canonical-link theorem or an explicit "standalone" classification.

```sh
cd proof/corpus && lake build <module set fixed in the spec> && lake build AxiomAudit
```

### Invariants

1. No canonical source moves and no pin change: `../sanguine-apeiron/proof` and the bridge manifest
   stay as they are.
2. Statement fidelity: a rebased headline keeps its historical statement, or carries a Sol-reviewed
   change note; a changed statement is renamed and its grade re-earned, never carried.
3. Historical artifacts are references with recorded digests, not acceptance.
4. Kernel discipline: no `sorry`/`axiom`/`admit`, no `set_option maxHeartbeats 0`, no whole-source
   `decide`/`rfl`, no raised `maxRecDepth`; bounded runs under the lane lock.
5. One writer per file; the `lakefile.lean` root list and `AxiomAudit.lean` are single-writer at a
   time.
6. No performance or publication claim: measurement is stage 6, release waits for rebase plus
   end-to-end proof.

### Verification hypotheses

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| Each admitted mechanism statement holds at the current identity | proved | focused `lake build <module>`; `#audit` per headline |
| The canonical-link theorem named in each row | proved | the row's named theorem |
| The rebased statement matches its historical source | anchor | per-row digest comparison and review note |
| The executing correspondence | proved (stage 3) | `proof/regen.sh --check` and the module gate |
| Learning-cost improvement at matched capability | measured (stage 6) | named machine, capability per FLOP against a pinned baseline |

### Non-goals

- New architectures or training runs; `AI-ARCH` is stage 6.
- Re-running the spinor-ttl measurements.
- The `NullCell` terrain/world modules and the physics, security and game rows.
- Publication; pre-rebase artifacts stay private.
- Editing the canonical corpus or the pinned package.
- Rust or extraction changes (stage 3).

### Trust boundary

The rebase claims cover Lean-kernel-checked statements for the admitted modules and the named links
between them and the canonical definitions. They do not cover the historical modules' own toolchain
behavior, rustc/LLVM codegen of a future extraction, GPU/SPIR-V, the learning dynamics that the
model statements abstract, or the instruments behind the historical measurements. The pinned corpus
identity `3d66ea3a…` is trusted for the canonical definitions; this epic does not audit it.

### Open questions

- Rebase home: new modules under `proof/corpus/Neural/` importing the pinned `apeiron` package (lead
  default; keeps the pin and the stage-3 extraction path) versus contributions to
  `../sanguine-apeiron`. Owner: lead; decide before the spec.
- First-wave module list and compile status under `v4.31.0`: the [inventory](neural-rebase.md)
  (anchor) records the per-file digests, imports and the wave proposal, and the
  [spec](neural-rebase.md) fixes the wave; compile status remains the focused-module gates.
  Owner: lead.
- Load-bearing set for the outcome row: `MirasCountLM`, `AttentionSpinor`, `NullTransformer`,
  `FrontierLadder` and `MLPrimitivesBridge` are candidates; whether the spec admits more of
  `Neural/` is a wave decision. Owner: lead.
- `SpinorFold` (`NullCell/`): stage 2 or the fold crossover. Owner: lead.
- Paper2's one-object/one-fold loop results: stage 2 or the Song correspondence (stage 4).
  Owner: lead.

## Specification — Neural rebase (verified-AI sequence, stage 2)

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/neural-rebase.md`](neural-rebase.md)
- **Inventory:** [`docs/epics/neural-rebase.md`](neural-rebase.md)
  (grade `anchor`; per-file digests, imports, wave proposal)
- **Source revision:** `74c037b`; working tree dirty (in-flight `Growable/Read.lean`,
  `Growable/RelocationWF.lean`; uncommitted `Growable/Classifier.lean`, `AxiomAudit.lean`, docs)
- **Status:** draft

```text
SCOPE: docs/epics/neural-rebase.md only; planning, no source, theorem or gate change
REGISTER: refinement planning for the AI-track outcome; the executable wave is stage 2 of the
          verified-AI sequence and produces no Rust or extraction change
CORPUS: canonical definitions Apeiron.Table.{HMap, get, insertWith, build} and
        Apeiron.Trichotomy.AnyRing.{RingCell, ringMul, ringNormForm} at the pinned corpus
        (../sanguine-apeiron/proof, corpus Git 3d66ea3aa2fe1de5997b6b757247454a2f771358);
        historical statements at ../sanguine/proof/Substrate/Algebra/Neural/ (per-file digests
        in the inventory)
REFERENCE: closure computed by recursive import scan (/tmp/opencode/neural-closure.py):
           46 modules, 16,371 lines; executed compile `lake build SubstrateVendored` finds three
           source-level failures (MadelungCell.lean:214, UniverseSyntax.lean:294,
           RadixDescent.lean:203/236/1157/1205) blocking 20 dependents; `import Std.Data.HashMap`
           resolves at v4.31.0 (`lake build Substrate.CollapseCore` exit 0), so no import adaptation
           is required
GATE: git diff --no-index --check /dev/null docs/epics/neural-rebase.md; no Lean gate by this edit
GRADE: anchor/planning; the wave's own module gates are the acceptance
```

### Scope and admissible domain

This specification re-establishes the historical Neural statement set under the current pinned
identity: `proof/corpus`, Lean `v4.31.0`, Mathlib `fabf563a…`. It specifies a vendoring-and-gate
wave, not a Rust operation and not an extraction (stage 3). The wave admits six modules; every other
vendored file enters only as required import closure and carries no headline claim.

| Item | Value |
|---|---|
| Operation | Vendor the admitted modules and their transitive import closure into `proof/corpus`, then compile and audit at the pinned identity |
| Domain | 6 admitted modules + closure = 46 modules, 16,371 lines (recursive `import` scan of `../sanguine/proof`, Mathlib/Batteries/Std excluded) |
| Totality | total Lean definitions; conditional theorems keep their hypotheses; no partial operation is specified |
| Preconditions | pinned toolchain and Mathlib; historical per-file sha256 recorded; lane lock and per-command budget |
| Postcondition | every wave row's focused `lake build <Module>` exits 0; every headline is registered in `AxiomAudit` inside `{propext, Classical.choice, Quot.sound}`; fidelity digests match except named adaptations |

#### Wave-1 rows

| # | Module (historical) | sha256 | Home (rebased) | Canonical link | Gate |
|---|---|---|---|---|---|
| 1 | `Substrate/Algebra/Neural/MirasCountLM.lean` | `9738ee17…` | `proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean` | standalone; count-table link deferred to stage 3 | `lake build Substrate.Algebra.Neural.MirasCountLM` |
| 2 | `Substrate/Algebra/Neural/KneserNey.lean` | `4611c407…` | same tree | standalone | `lake build Substrate.Algebra.Neural.KneserNey` |
| 3 | `Substrate/Algebra/Neural/CountLMFloor.lean` | `d82351a7…` | same tree | standalone; carries `refine_gap_zero_of_kappa_sufficient` with its undischarged antecedent (tracked by `FrontierLadder`, wave 2) | `lake build Substrate.Algebra.Neural.CountLMFloor` |
| 4 | `Substrate/Algebra/Neural/AttentionSpinor.lean` | `7cb47e7d…` | same tree | standalone; O(1)-state scan mechanism | `lake build Substrate.Algebra.Neural.AttentionSpinor` |
| 5 | `Substrate/Algebra/Neural/MLPrimitivesBridge.lean` | `32bba81f…` | same tree | standalone; paper1-named headlines (`osFuse_*`, quant dual) | `lake build Substrate.Algebra.Neural.MLPrimitivesBridge` |
| 6 | `Substrate/Algebra/Neural/TrainingCollapse.lean` | `9eeb8663…` | same tree | standalone; `ridgeHead_unique`, `gram_stack` | `lake build Substrate.Algebra.Neural.TrainingCollapse` |

Deferred to wave 2, recorded so they are not dropped: `NullCell/SpinorFold.lean` (`b5c274fa…`;
fold-crossover decision), `Neural/FrontierLadder.lean` (`cad186b0…`; ledger over wave-1 modules),
`Neural/NullTransformer.lean` (`636ba26a…`; composite consumer, 1,648 lines and over the 1,500-line
file limit — split required first).

### Representation relation (historical to rebased)

| Layer | Symbol | Path |
|---|---|---|
| Historical source | each admitted module | `/home/socol/Workspace/sanguine/proof/…`, sha256 per inventory |
| Rebased module | same module name | `proof/corpus/Substrate/…` (mirrored tree, imports unchanged) |
| Canonical definition | `Apeiron.Table.*`, `Apeiron.Trichotomy.AnyRing.*` | pinned corpus; link named per row, standalone in wave 1 |
| Fidelity check | vendored bytes | `sha256sum -c proof/corpus/Substrate/vendor.sha256` |
| Kernel link | headline registration | `proof/corpus/AxiomAudit.lean` |

Rules:

1. A vendored file is copied byte-identical; module names and `import` lines are unchanged, so the
   old tree structure is mirrored rather than rewritten.
2. No import adaptation is required: `Substrate/CollapseCore.lean:1` `import Std.Data.HashMap`
   resolves at the pinned `v4.31.0` (`lake build Substrate.CollapseCore` exit 0; toolchain module
   `Std/Data/HashMap.lean`). The earlier "absent module" claim was a name-prefix inference; the
   executed compile corrects it.
3. Compile-level adaptations are named in the plan row before the edit and are minimal:
   `UniverseSyntax.lean` (unused variable `prev`, linter-as-error), `RadixDescent.lean` (`push_neg`
   and `Finset.filter_card_add_filter_neg_card_eq_card` deprecations, unused variable `hb`), and
   `MadelungCell.lean` (proof term at 214:2 under the pinned Mathlib; statement unchanged). Each
   adapted file is listed separately with its new digest, is excluded from the byte-identical set,
   and is reproduced on a vendor re-run through `tools/neural-adapt/*.patch`. If a declaration name
   or statement must change, the row records the change and an independent verifier verdict before
   closure, and the old name is retired explicitly. Any further adaptation follows the same rule.
4. No vendored file may exceed 1,500 lines or push a directory over 12 immediate source files
   (AGENTS.md §4). The closure maximum is 1,386 lines (`Closures/Engineering/RadixDescent`) and the
   largest vendored directory is 10 files (`Substrate/Algebra/Bulk`), so no split is required for
   wave 1.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `closure_complete` | every `import` of every vendored module resolves in the vendored tree or the pinned dependency closure (Mathlib, Batteries, Std) | proved | per-row focused `lake build`; the wave barrier `lake build` |
| `fidelity` | for every vendored file except the listed adaptations, sha256 equals the historical digest | anchor | `sha256sum -c proof/corpus/Substrate/vendor.sha256` |
| `kernel_cone` | every wave headline's transitive axioms are inside `{propext, Classical.choice, Quot.sound}` | proved | `#audit` lines in `AxiomAudit.lean`; `lake build AxiomAudit` |
| `no_sorry` | no `sorry`, `admit` or `axiom` declaration in the vendored tree | proved | anchored `rg '^\s*(sorry\|admit\|axiom)\b'` scan; AxiomAudit build |
| `statement_fidelity` | for the adapted file, declaration names and statements equal the historical ones | anchor | name/statement diff for that file, recorded in the plan row |
| `limits` | every vendored file ≤1,500 lines; every vendored directory ≤12 immediate files | anchor | `wc -l` and file-count check in the wave gate |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Import resolves to no module at `v4.31.0` | task fails; the adaptation is recorded in the plan row | none in this closure; `Std.Data.HashMap` resolves (rule 2) |
| Declaration renamed in Mathlib under an unchanged module path | task fails at the focused gate; adaptation recorded | spec § rules 3-4 |
| File over 1,500 lines or directory over 12 files | wave rejects the row; a split task precedes it | AGENTS.md §4 |
| `sorry`/`axiom` present | `AxiomAudit` build failure | closed cone |
| Any statement change without a plan row | task rejected | invariant 2 of the intent |

### Evidence plan

1. **Positive.** Per row, `cd proof/corpus && lake build <Module>` under the lane lock; wave barrier
   runs `lake build` and `lake build AxiomAudit`.
2. **Proof.** The Lean kernel `lake build` of the vendored modules and of `AxiomAudit`.
3. **Axiom cone.** `#audit` registration for the wave headlines; anything outside
   `{propext, Classical.choice, Quot.sound}` fails the build.
4. **Negative controls.** (a) Fidelity: flip one byte in a disposable vendored copy and require
   `sha256sum -c` to fail; (b) statement: mutate one wave-1 headline in a disposable copy (candidate:
   `MirasCountLM.forget_hebb`, whose statement is an exact inverse) and require its proof to fail
   with an attributable diagnostic; restore the original bytes and re-run green.
5. **Measurement.** None; this wave makes no performance claim.

### Trust boundary

The wave claims cover the kernel-checked statements of the vendored modules at the pinned identity,
the digest fidelity of the copies to the recorded historical files, and the closed axiom cone. It
does not cover: the historical corpus beyond those digests; the canonical corpus pin itself; Rust,
Aeneas, LLVM or any executing path (stage 3); the `../spinor-ttl` measurements and their instruments;
training dynamics; GPU/SPIR-V; or publication. Wave-1 rows are `standalone`: they establish the
mechanism statements, not their link to the canonical `Apeiron.Table`/`Trichotomy` definitions. That
link is a later wave and a separate claim.

### Open questions

- Root mechanism (resolved, W0-T2): per-subtree `globs` in a second `@[default_target] lean_lib`
  (`SubstrateVendored`) owns the vendored modules; `ApeironWordBridge` keeps the explicit root list
  and does not import the vendored modules yet.
- Wave-2 split of `NullTransformer` (1,648 lines): which modules carry the split.
- `SpinorFold`: stage 2 wave 2 or the fold-crossover item.
- Whether the closure-only `Substrate.Algebra.Translation.*` and `Substrate.Algebra.Security.*`
  modules remain under `proof/corpus/Substrate/` when stage 4 opens.

## Plan — Neural rebase (verified-AI sequence, stage 2)

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/neural-rebase.md`](neural-rebase.md)
- **Spec:** [`docs/epics/neural-rebase.md`](neural-rebase.md)
- **Inventory:** [`docs/epics/neural-rebase.md`](neural-rebase.md)
- **Source revision:** `74c037b`; working tree dirty (in-flight `Growable/Read.lean`,
  `Growable/RelocationWF.lean` and `proof/corpus/lakefile.lean`; uncommitted `Growable/Classifier.lean`,
  `AxiomAudit.lean`, docs)
- **Status:** draft

```text
SCOPE: docs/epics/neural-rebase.md only; planning, no source, theorem or gate change
REGISTER: refinement planning for the AI-track outcome (docs/direction.md#programme-outcomes,
          "Active test-time learning and lower-cost useful AI"); excludes execution, grade changes,
          pin changes and performance work
CORPUS: canonical definitions Apeiron.Table.{HMap, get, insertWith, build} and
        Apeiron.Trichotomy.AnyRing.{RingCell, ringMul, ringNormForm} at the pinned corpus
        (../sanguine-apeiron/proof, corpus Git 3d66ea3aa2fe1de5997b6b757247454a2f771358 per the
        bridge manifest); historical statements at ../sanguine/proof/Substrate/Algebra/Neural/ and
        ../sanguine/proof/Substrate/Algebra/NullCell/SpinorFold.lean
REFERENCE: docs/epics/neural-rebase.md; docs/epics/neural-rebase.md; docs/epics/neural-rebase.md;
           docs/roadmap.md ("Verified-AI sequence and prior-art index");
           docs/epics/agent-throughput-segmentation.md; /tmp/opencode/neural-closure.py
GATE: git diff --no-index --check /dev/null docs/epics/neural-rebase.md; cited path resolution check
GRADE: anchor/planning; historical grades retained, not fresh acceptance
```

### 1. Goal and completion condition

Rebase the six admitted neural modules and their transitive import closure (46 modules, 16,371 lines)
from the historical Sanguine corpus into `proof/corpus/Substrate/` under Lean `v4.31.0` and Mathlib
`fabf563a…`. Wave 1 establishes focused exit 0 for each admitted module and registers every headline
in `AxiomAudit.lean` inside `{propext, Classical.choice, Quot.sound}`. Wave 2 addresses deferred
downstream modules, line-limit decomposition, and decision items.

```sh
cd proof/corpus && lake build Substrate.Algebra.Neural.MirasCountLM \
  Substrate.Algebra.Neural.KneserNey \
  Substrate.Algebra.Neural.CountLMFloor \
  Substrate.Algebra.Neural.AttentionSpinor \
  Substrate.Algebra.Neural.MLPrimitivesBridge \
  Substrate.Algebra.Neural.TrainingCollapse \
  && lake build AxiomAudit
```

### 2. Non-goals and trust boundary

- Non-goals: Rust runtime changes or Aeneas extraction (stage 3); model training or benchmark
  measurements (stage 6); modifying the pinned canonical corpus at `../sanguine-apeiron/proof`;
  NullCell world/terrain modules; publication.
- Trust boundary: covers the kernel-checked statements of the vendored modules at Lean `v4.31.0` and
  Mathlib `fabf563a…`, file digest fidelity against historical sources, and the closed axiom cone.
  Does not cover historical toolchains, future rustc/LLVM codegen, hardware behavior, or historical
  measurement instruments. Wave-1 modules are standalone; links to canonical `Apeiron.Table` are
  stage-3 obligations.

### 3. Lane map and serialization rules

Per `docs/epics/agent-throughput-segmentation.md`:

1. **Integrator lane (lead):** owns `proof/corpus/lakefile.lean` (root declarations) and
   `proof/corpus/AxiomAudit.lean` (headline audit registration). Only the integrator edits these
   files; worker tasks never touch them.
2. **Worker lane (Flash / default):** one writer per file. Each module rebase task owns exclusively
   its single vendored `.lean` file. Tasks within a wave execute on disjoint files.
3. **Execution serialization:** all Lean commands run under the lane lock:
   `flock /tmp/opencode/apeiron-lane.lock` under bounded memory and swap
   (`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0` with `LEAN_NUM_THREADS=1`).
4. **Command and task budgets:** 10-minute ceiling per command (AGENTS.md §10); 45-minute strict
   budget per task. Expiry halts execution and reports the blocker.

### 4. Wave map

| Wave | Task ID | Owner | Owned files (≤3) | One focused gate | Negative control |
|---|---|---|---|---|---|
| W0 | `APEIRON-NEURAL-W0-T1` | integrator | `tools/vendor-neural.sh`, `proof/corpus/Substrate/vendor.sha256` | `bash tools/vendor-neural.sh && sha256sum -c proof/corpus/Substrate/vendor.sha256` | Flip 1 byte in disposable vendored copy; `sha256sum -c` must fail |
| W0 | `APEIRON-NEURAL-W0-T2` | integrator | `proof/corpus/lakefile.lean` | `cd proof/corpus && lake build ApeironWordBridge` | Add invalid root in disposable copy; `lake build` must fail |
| W0 | `APEIRON-NEURAL-W0-T3` | integrator | `tools/vendor-neural.sh`, `proof/corpus/Substrate/adapted.sha256` | `bash tools/vendor-neural.sh && sha256sum -c proof/corpus/Substrate/adapted.sha256` | Byte-flip in a scratch copy; adapted check fails |
| W1 | `APEIRON-NEURAL-W1-T0` | — | retired: `import Std.Data.HashMap` resolves; `lake build Substrate.CollapseCore` exit 0 | — | none required |
| W1 | `APEIRON-NEURAL-W1-T0A` | worker | `proof/corpus/Substrate/Algebra/Translation/UniverseSyntax.lean` | `cd proof/corpus && lake build Substrate.Algebra.Translation.UniverseSyntax` | Revert the rename in a disposable copy; unused-variable diagnostic |
| W1 | `APEIRON-NEURAL-W1-T0B` | worker | `proof/corpus/Closures/Engineering/RadixDescent.lean` | `cd proof/corpus && lake build Closures.Engineering.RadixDescent` | Revert a deprecated-name rename in a disposable copy; deprecation diagnostic |
| W1 | `APEIRON-NEURAL-W1-T0C` | worker | `proof/corpus/Substrate/Algebra/QuantumField/MadelungCell.lean` | `cd proof/corpus && lake build Substrate.Algebra.QuantumField.MadelungCell` | Revert the proof repair in a disposable copy; type-mismatch diagnostic |
| W1 | `APEIRON-NEURAL-W1-T0D` | worker | `proof/corpus/Closures/Engineering/CoordinateReadout.lean` | `cd proof/corpus && lake build Closures.Engineering.CoordinateReadout` | Revert the `omit`/`push Not` edit in a disposable copy; linter/deprecation diagnostic |
| W1 | `APEIRON-NEURAL-W1-T0E` | worker | `proof/corpus/Substrate/Algebra/Neural/SparseAttentionBudget.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.SparseAttentionBudget` | Revert the rename in a disposable copy; unused-variable diagnostic |
| W1 | `APEIRON-NEURAL-W1-T1` | worker | `proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.MirasCountLM` | Mutate `forget_hebb` statement in disposable copy; proof fails |
| W1 | `APEIRON-NEURAL-W1-T2` | worker | `proof/corpus/Substrate/Algebra/Neural/KneserNey.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.KneserNey` | Mutate `kn_normalizes` statement in disposable copy; proof fails |
| W1 | `APEIRON-NEURAL-W1-T3` | worker | `proof/corpus/Substrate/Algebra/Neural/CountLMFloor.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.CountLMFloor` | Mutate `refine_gap_zero_of_kappa_sufficient` statement in disposable copy; proof fails |
| W1 | `APEIRON-NEURAL-W1-T4` | worker | `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.AttentionSpinor` | Mutate `attention_is_spinor_scan` statement in disposable copy; proof fails |
| W1 | `APEIRON-NEURAL-W1-T5` | worker | `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.MLPrimitivesBridge` | Mutate `osFuse_is_add` statement in disposable copy; proof fails |
| W1 | `APEIRON-NEURAL-W1-T6` | worker | `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.TrainingCollapse` | Mutate `ridgeHead_unique` statement in disposable copy; proof fails |
| W1 | `APEIRON-NEURAL-W1-T7` | integrator | `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build AxiomAudit` | Seed unapproved axiom into audit list in disposable copy; audit fails |
| W2 | `APEIRON-NEURAL-W2-T1` | worker | `proof/corpus/Substrate/Algebra/Neural/NullTransformer/Core.lean`, `proof/corpus/Substrate/Algebra/Neural/NullTransformer/Layers.lean`, `proof/corpus/Substrate/Algebra/Neural/NullTransformer.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.NullTransformer` | Mutate block forward composition in disposable copy; build fails |
| W2 | `APEIRON-NEURAL-W2-T2` | worker | `proof/corpus/Substrate/Algebra/Neural/FrontierLadder.lean` | `cd proof/corpus && lake build Substrate.Algebra.Neural.FrontierLadder` | Mutate rung reference in disposable copy; build fails |
| W2 | `APEIRON-NEURAL-W2-T3` | integrator | `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean` | `cd proof/corpus && lake build Substrate.Algebra.NullCell.SpinorFold` | Mutate `nmul_assoc` statement in disposable copy; proof fails |
| W2 | `APEIRON-NEURAL-W2-T4` | integrator | `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build AxiomAudit` | Seed unapproved axiom for W2 headlines; audit fails |

### 5. Task blocks

#### APEIRON-NEURAL-W0-T1 · Deterministic vendor script and manifest

- **Scope:** Create `tools/vendor-neural.sh` and populate `proof/corpus/Substrate/vendor.sha256`
  covering the 46 modules identified in `/tmp/opencode/neural-closure.py`.
- **Files:** `tools/vendor-neural.sh`, `proof/corpus/Substrate/vendor.sha256`
- **Depends on:** none
- **Grade:** `anchor` (reproducible copy and digest verification)
- **Gate:**
  ```sh
  bash tools/vendor-neural.sh && sha256sum -c proof/corpus/Substrate/vendor.sha256
  ```
- **Negative control:** Invert one byte in a vendored file in a disposable copy; `sha256sum -c` must
  exit nonzero with a checksum mismatch diagnostic. Restore file and verify exit 0.
- **Evidence to record:** Script source, recorded `vendor.sha256` content, execution log.

#### APEIRON-NEURAL-W0-T2 · Lakefile root mechanism configuration

- **Scope:** Configure root discovery for vendored subtrees in `proof/corpus/lakefile.lean`. Test
  Lake 5 `globs` mechanism; fallback to explicit roots if `globs` fails under the pinned toolchain.
- **Files:** `proof/corpus/lakefile.lean`
- **Depends on:** `APEIRON-NEURAL-W0-T1`
- **Grade:** `tested`
- **Gate:**
  ```sh
  cd proof/corpus && lake build ApeironWordBridge
  ```
- **Negative control:** Introduce a syntax error or nonexistent root in `proof/corpus/lakefile.lean`
  in a disposable copy; `lake build` must exit nonzero. Restore and verify exit 0.
- **Evidence to record:** Chosen root configuration diff and build output.

**Result (adopted).** Option A as a second default target:

```lean
@[default_target]
lean_lib SubstrateVendored where
  globs := #[.submodules `Substrate, .submodules `Closures]
```

- `lake build ApeironWordBridge` exits 0 (3821 jobs); `lakefile.lean` sha256
  `586fba16c5668d0222376f9f00b620c197bd94b1ac7ab660de19dc71ea9fcbdc`; log
  `/tmp/opencode/w0t2-gate-apeironwordbridge.log`. `ApeironWordBridge` is unchanged and does not
  import the vendored modules yet; the vendored lib is inside bare `lake build` for the wave barriers.
- Mechanism positive: the globs lib resolves every manifest module — `lake build SubstrateVendored`
  attempts all 46 with no `unknown target`, building 23 and failing 3 at source sites:
  `Substrate/Algebra/QuantumField/MadelungCell.lean:214:2` (HasDerivAt type mismatch after
  simplification), `Substrate/Algebra/Translation/UniverseSyntax.lean:294:30` (unused variable
  `prev`), `Closures/Engineering/RadixDescent.lean:203/236/1157/1205` (`push_neg` and
  `Finset.filter_card_add_filter_neg_card_eq_card` deprecations, unused variable `hb`). 20 further
  modules have no olean because they depend on the three; the wave-1 adaptations below are the
  unblocking work. Log `/tmp/opencode/w0t2-vendored-full.log`.
- Control: scratch package under the pinned toolchain with the same two-lib pattern and a nested
  vendor tree (`Vendor/B/C`); baseline builds, a missing-glob mutation fails with a `job computation`
  failure, restore passes (0/1/0; `/tmp/opencode/w0t2-control/`). Pattern-level, not a full corpus
  copy.

#### APEIRON-NEURAL-W0-T3 · Adaptation-aware vendor re-run and final-tree manifest (integrator)

- **Scope:** Extend `tools/vendor-neural.sh` so a re-run reproduces the adapted tree: copy the 46
  historical files, verify the copies against `Substrate/vendor.sha256`, apply
  `tools/neural-adapt/*.patch` in sorted order, then verify the final tree against
  `Substrate/adapted.sha256`. Generate the initial `adapted.sha256` from the current tree (no
  adaptations yet).
- **Files:** `tools/vendor-neural.sh`, `proof/corpus/Substrate/adapted.sha256`
- **Depends on:** `APEIRON-NEURAL-W0-T1`
- **Grade:** `tested`
- **Gate:**
  ```sh
  bash tools/vendor-neural.sh && sha256sum -c proof/corpus/Substrate/adapted.sha256
  ```
- **Negative control:** In a scratch copy under the repo-root layout, flip one byte of a vendored file
  and require the adapted check to fail with a checksum mismatch; restore and verify exit 0.
- **Evidence to record:** Script diff, manifest, control logs, lakefile-independent.

#### APEIRON-NEURAL-W1-T0 · CollapseCore import adaptation — RETIRED (verified inapplicable)

- **Verified:** `import Std.Data.HashMap` resolves at the pinned `v4.31.0` toolchain (module
  `Std/Data/HashMap.lean`); `cd proof/corpus && lake build Substrate.CollapseCore` exits 0 (442 ms,
  log `/tmp/opencode/w0t2-probe-collapsecore.log`). No edit; the file remains byte-identical and
  inside `vendor.sha256`.
- **Correction:** the closure receipt's unresolved-import list was a name-prefix inference; the
  executed compile is the gate and it is green. Retired without adaptation; excluded population
  none.
- **Replacement work:** the compile-level adaptations the vendored build found are W1-T0A/B/C below.

#### APEIRON-NEURAL-W1-T0A · UniverseSyntax linter adaptation

- **Scope:** `Substrate/Algebra/Translation/UniverseSyntax.lean:294:30` reports `Variable name
  `prev` is not explicitly referenced.` under the package's `warningAsError`. Rename the binding
  (statement-preserving: no declaration name or type changes).
- **Files:** `proof/corpus/Substrate/Algebra/Translation/UniverseSyntax.lean`
- **Depends on:** `APEIRON-NEURAL-W0-T2`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Translation.UniverseSyntax
  ```
- **Negative control:** In a disposable copy, revert the rename; the focused build must fail with the
  unused-variable diagnostic. Restore and verify exit 0.

**Result (closed).** `prev` -> `_prev` at `UniverseSyntax.lean:294`; adapted digest
`051cb5e55111200d02d83d7008cd9ba9cd80b2e1fe1ff886797c3d912b95b630`; patch
`tools/neural-adapt/UniverseSyntax.patch` (`b45c2c0e…`). Gate exit 0 (3 jobs) before and after the
vendor re-run; `bash tools/vendor-neural.sh && sha256sum -c proof/corpus/Substrate/adapted.sha256`
exit 0 (46 OK, patch applied); control baseline/red/restore 0/1/0 with the 294:30 diagnostic in red.
`vendor.sha256` unchanged (`ebfa7ccb…`).

#### APEIRON-NEURAL-W1-T0B · RadixDescent deprecation adaptation

- **Scope:** `Closures/Engineering/RadixDescent.lean` at 203:2 (`push_neg` -> `push Not`), 236:30
  (unused variable `hb` -> `_hb`), 1157:12 and 1205:14
  (`Finset.filter_card_add_filter_neg_card_eq_card` -> `Finset.card_filter_add_card_filter_not`; the
  deprecation message's shorter suggestion does not exist at this pin). Statement-preserving renames
  only.
- **Files:** `proof/corpus/Closures/Engineering/RadixDescent.lean`
- **Depends on:** `APEIRON-NEURAL-W0-T2`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Closures.Engineering.RadixDescent
  ```
- **Negative control:** In a disposable copy, revert one rename; the focused build must fail with the
  deprecation diagnostic. Restore and verify exit 0.

**Result (closed).** Four statement-preserving edits; adapted digest
`027069b9b90cc32a292b8387ab1905dc04997fe94dd28ec3fcea94eeb8e21891`; patch
`tools/neural-adapt/RadixDescent.patch` (`91fc060e…`). Gate exit 0 (1204 jobs) before and after the
vendor re-run; the vendor re-run applies both patches and verifies `adapted.sha256` (`5f3820a7…`,
46 OK); control baseline/red/restore 0/1/0 with all four diagnostics in red at 203:2, 236:30,
1157:12, 1205:14. The compiler-defined replacement name is `Finset.card_filter_add_card_filter_not`.

#### APEIRON-NEURAL-W1-T0C · MadelungCell proof repair under the pinned Mathlib

- **Scope:** `Substrate/Algebra/QuantumField/MadelungCell.lean:214:2` fails with a `HasDerivAt` type
  mismatch after simplification (`(R * R) y * ddφ y` vs `R y * R y * ddφ y`). Repair the proof term
  at the same statement; no declaration name or statement change. If a statement change proves
  necessary, stop and record it for an independent verifier verdict before the edit.
- **Files:** `proof/corpus/Substrate/Algebra/QuantumField/MadelungCell.lean`
- **Depends on:** `APEIRON-NEURAL-W0-T2`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.QuantumField.MadelungCell
  ```
- **Negative control:** In a disposable copy, revert the repaired proof term; the focused build must
  fail with the type-mismatch diagnostic. Restore and verify exit 0.

**Result (closed).** Proof repaired at `MadelungCell.lean:214` by `change HasDerivAt (R * R * dφ) _ _`
then `simpa only [Pi.mul_apply] using ((hR y).mul (hR y)).mul (hdφ y)`; adapted digest
`bd2075421135a4f436836c404ebe6f90067342eb2191a1acad8e909fd9bdb953`; patch
`tools/neural-adapt/MadelungCell.patch` (`98148b71…`). Gate exit 0 (2110 jobs) before and after the
vendor re-run; the vendor re-run applies all three patches and verifies `adapted.sha256` (`981892e2…`,
46 OK); control baseline/historical/restore 0/1/0 with the 214:2 type-mismatch diagnostic in red.

**Follow-on (`lake build SubstrateVendored`, all three patches).** 41 of 46 modules build. Two new
root failures remain: `Substrate/Algebra/Neural/TrainingCollapse.lean` (197:4, 589:4, 651:4 type
mismatch; 573:59, 573:68, 574:8, 722:4, 722:32, 722:54, 722:77, 724:21 unused simp arguments) and
`Closures/Engineering/CoordinateReadout.lean` (116:0 unused section variable `[DecidableEq K]`;
402:2 `push_neg` deprecation), blocking the three remaining unbuilt modules. Handled by `W1-T6`
(extended) and the new `W1-T0D`.

All three adaptations: record the diff against the historical file, the new sha256 in
`Substrate/adapted.sha256` (regenerate from the repository root:
`awk '{print $2}' proof/corpus/Substrate/vendor.sha256 | xargs sha256sum >
proof/corpus/Substrate/adapted.sha256`), and the `tools/neural-adapt/` patch entry —
`diff -u --label a/proof/corpus/<rel> --label b/proof/corpus/<rel>
/home/socol/Workspace/sanguine/proof/<rel> proof/corpus/<rel> > tools/neural-adapt/<name>.patch`,
applied by `tools/vendor-neural.sh` with `git apply` from the repository root. The manifest
regeneration and vendor re-run are serialized on the integrator lane; a worker records only its
patch.

#### APEIRON-NEURAL-W1-T0D · CoordinateReadout linter adaptation

- **Scope:** `Closures/Engineering/CoordinateReadout.lean`: 116:0 unused section variable
  `[DecidableEq K]` in `partial_tag_false_positive` (add `omit [DecidableEq K] in`) and 402:2
  `push_neg` -> `push Not`. Statement-preserving.
- **Files:** `proof/corpus/Closures/Engineering/CoordinateReadout.lean`,
  `tools/neural-adapt/CoordinateReadout.patch`
- **Depends on:** `APEIRON-NEURAL-W1-T0B`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Closures.Engineering.CoordinateReadout
  ```
- **Negative control:** In a disposable copy, revert one edit; the focused build must fail with the
  linter/deprecation diagnostic. Restore and verify exit 0.

**Status.** Adapted digest `dd4c7570353ff6d5e28ab75d15185440fe5d608905b1c7b163a9a19ab9d95cbb`; patch
`tools/neural-adapt/CoordinateReadout.patch` (`087c333b…`); gate exit 0 (2222 jobs); control
historical/adapted/restore 1/0/0 with diagnostics at `116:0` and `402:2` in red. `omit [DecidableEq K]
in` must precede the doc comment (Lean rejects it between the doc comment and `theorem`; reproduced on
a scratch file); the declaration name, type and statement are unchanged. Manifest regeneration and the
vendor re-run are serialized with `W1-T6` (pending).

#### APEIRON-NEURAL-W1-T0E · SparseAttentionBudget linter adaptation

- **Scope:** `Substrate/Algebra/Neural/SparseAttentionBudget.lean`: 80:51 unused `hp`, 103:28
  unused `hn`; rename `_hp`/`_hn`, statement-preserving. The module is in the import closure of
  `CountLMFloor`.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/SparseAttentionBudget.lean`,
  `tools/neural-adapt/SparseAttentionBudget.patch`
- **Depends on:** `APEIRON-NEURAL-W0-T2`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.SparseAttentionBudget
  ```
- **Negative control:** In a disposable copy, revert a rename; the focused build must fail with the
  unused-variable diagnostic. Restore and verify exit 0.

**Result (closed).** `_hp` (80:51; the use inside the nested `Finset.sum_le_sum` block renamed with it)
and `_hn` (103:28); adapted digest
`978bbde9e0b2d06e646ecc02840b4816eb3ffa113335146d49d5b9ec170a5d02`; patch
`tools/neural-adapt/SparseAttentionBudget.patch` (`c093f368…`). Gate exit 0 (3127 jobs); control
historical red at 80:51/103:28, adapted green; the patch reproduces the adapted file from the
historical copy byte-identically.

#### APEIRON-NEURAL-W1-T1 · Rebase MirasCountLM

- **Scope:** Verify compilation of `Substrate.Algebra.Neural.MirasCountLM` at `v4.31.0`. Headlines:
  `hebb_comm`, `forget_hebb`, `hebb_forget`, `surprise_eq_info`, `countlm_is_exact_miras`.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean`
- **Depends on:** `APEIRON-NEURAL-W1-T0`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.MirasCountLM
  ```
- **Negative control:** Statement control: in a disposable copy, mutate `forget_hebb` theorem
  conclusion from `= M` to `= hebb obs M`; `lake build` must exit nonzero with tactic failure.
  Restore and verify exit 0.
- **Evidence to record:** Compilation log, sha256 equality with historical file `9738ee17…`.

**Result (closed).** Gate exit 0 (2970 jobs). Control: baseline 0, red 1 with
`MirasCountLM.lean:101:87` unsolved goal and the `countlm_is_exact_miras` consumer type mismatch at
`142:20`, restore 0; live digest `9738ee17f7f7d130f69ab6c345743e7c2c06517d624a27d0cb4c7e6dd359577e`
equals the historical digest (unadapted); `ApeironWord.lean` unchanged.

#### APEIRON-NEURAL-W1-T2 · Rebase KneserNey

- **Scope:** Verify compilation of `Substrate.Algebra.Neural.KneserNey` at `v4.31.0`. Headlines:
  `kn_normalizes`, `kn_lower_order_weight_bound`, `kn_density_support`.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/KneserNey.lean`
- **Depends on:** `APEIRON-NEURAL-W0-T2`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.KneserNey
  ```
- **Negative control:** In a disposable copy, mutate `kn_normalizes` conclusion from `= 1` to `= 0`;
  `lake build` must exit nonzero with proof failure. Restore and verify exit 0.
- **Evidence to record:** Compilation log, sha256 equality with historical file `4611c407…`.

**Result (closed).** Gate exit 0 (2966 jobs). Control: baseline 0, red 1 — the mutated conclusion `= 0`
at line 100 is rejected at the proof's `div_self` (`108:2`, expected `... = 0`), restore 0; live digest
`4611c407…` equals the historical digest.

#### APEIRON-NEURAL-W1-T3 · Rebase CountLMFloor

- **Scope:** Verify compilation of `Substrate.Algebra.Neural.CountLMFloor` at `v4.31.0`. Headlines:
  `refine_gap_zero_of_kappa_sufficient`, `refine_gap_zero_of_kappa_sufficient`, `entropy_lower_bound`.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/CountLMFloor.lean`
- **Depends on:** `APEIRON-NEURAL-W0-T2`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.CountLMFloor
  ```
- **Negative control:** In a disposable copy, mutate `refine_gap_zero_of_kappa_sufficient` conclusion in a theorem
  statement; `lake build` must exit nonzero. Restore and verify exit 0.
- **Evidence to record:** Compilation log, sha256 equality with historical file `d82351a7…`.

#### APEIRON-NEURAL-W1-T4 · Rebase AttentionSpinor

- **Scope:** Verify compilation of `Substrate.Algebra.Neural.AttentionSpinor` at `v4.31.0`.
  Headlines: `attention_is_spinor_scan`, `spinor_scan_preserves_norm`, `online_softmax_exact`.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/AttentionSpinor.lean`
- **Depends on:** `APEIRON-NEURAL-W0-T2`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.AttentionSpinor
  ```
- **Negative control:** In a disposable copy, mutate `attention_is_spinor_scan` formula; `lake build` must
  exit nonzero with proof failure. Restore and verify exit 0.
- **Evidence to record:** Compilation log, sha256 equality with historical file `7cb47e7d…`.

**Result (closed).** Gate exit 0. Control: baseline 0, red 1 — the mutated `+` -> `-` conclusion of
`attention_is_spinor_scan` is rejected at `126:3` through `attnStep_compose`, restore 0; live digest
`7cb47e7d…` equals the historical digest.

#### APEIRON-NEURAL-W1-T5 · Rebase MLPrimitivesBridge

- **Scope:** Rebase `Substrate.Algebra.Neural.MLPrimitivesBridge` at `v4.31.0`: adapt the 387:5
  unused-variable diagnostic (`hD0`) statement-preservingly, then verify. Headlines: `osFuse_is_add`,
  `osFuse_grouping_free`, `output_error_is_function_distortion`, `aware_scale_is_the_chart`,
  `weight_only_is_dual_blind`, `kn_floor_refinement`.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean`,
  `tools/neural-adapt/MLPrimitivesBridge.patch`
- **Depends on:** `APEIRON-NEURAL-W0-T2`, `APEIRON-NEURAL-W1-T0A/B/C`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.MLPrimitivesBridge
  ```
- **Negative control:** In a disposable copy, mutate `osFuse_is_add` statement; `lake build` must
  exit nonzero with proof failure. Restore and verify exit 0.
- **Evidence to record:** Compilation log, patch, adapted digest; historical file `32bba81f…`.

**Result (closed).** `_hD0` rename at 387:5; adapted digest
`bb2138087e995ad0d2f49a53e3db7f19ebbf380e9a625ff08c2b16eaf30598d4`; patch
`tools/neural-adapt/MLPrimitivesBridge.patch` (`7c064ea2…`). Gate exit 0; controls: historical red at
387:5, `osFuse_is_add` statement mutant red (`+`→`-` at 174:80), adapted copy green; live digest
verified after the vendor re-run.

#### APEIRON-NEURAL-W1-T6 · Rebase TrainingCollapse

- **Scope:** Rebase `Substrate.Algebra.Neural.TrainingCollapse` at `v4.31.0`: adapt the eleven
  compile diagnostics found by `lake build SubstrateVendored` — type mismatches at 197:4, 589:4,
  651:4 and unused simp arguments at 573:59, 573:68, 574:8, 722:4, 722:32, 722:54, 722:77, 724:21 —
  statement-preserving, then verify. Headlines: `ridgeHead_unique`, `gram_stack`,
  `normal_equations_stream`.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean`,
  `tools/neural-adapt/TrainingCollapse.patch`
- **Depends on:** `APEIRON-NEURAL-W0-T2`, `APEIRON-NEURAL-W1-T0A/B/C`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.TrainingCollapse
  ```
- **Negative control:** In a disposable copy, mutate `ridgeHead_unique` statement; `lake build` must
  exit nonzero with proof failure. Restore and verify exit 0.
- **Evidence to record:** Compilation log, sha256 equality with historical file `9eeb8663…`.

**Result (closed).** Eleven diagnostics adapted statement-preservingly (`simpa only` and unused
simp-argument cleanup); adapted digest
`329d8f674745f08af81276704b7b4e739924a099c60c658d1e9f128a50e7d81a`; patch
`tools/neural-adapt/TrainingCollapse.patch` (`db57b35a…`). Gate exit 0; control: historical red at the
eleven sites, `ridgeHead_unique` statement mutant red (`226:36`, `350:2`, `435:19`), adapted copy green.

#### APEIRON-NEURAL-W1-T7 · AxiomAudit registration for Wave 1

- **Scope:** Register Wave-1 headlines in `proof/corpus/AxiomAudit.lean` with `#audit` lines; verify
  axioms remain in `{propext, Classical.choice, Quot.sound}`.
- **Files:** `proof/corpus/AxiomAudit.lean`
- **Depends on:** `APEIRON-NEURAL-W1-T1` through `APEIRON-NEURAL-W1-T6`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build AxiomAudit
  ```
- **Negative control:** In a disposable copy, insert an `#audit` targeting an unproved declaration or
  axiom; `lake build AxiomAudit` must fail. Restore and verify exit 0.
- **Evidence to record:** List of registered headlines, `#audit` output, build status.

#### APEIRON-NEURAL-W2-T1 · Decomposition of the archived transformer composite (`Substrate/Algebra/Neural/NullTransformer.lean`)

- **Scope:** The historical composite `Substrate/Algebra/Neural/NullTransformer.lean` is 1,648 lines, exceeding the 1,500-line repository
  limit. Decompose by the resolved decision (`§7` 3): `Core.lean` carries the prefix (the archived
  §1 nonlinearity derivation, ops, node graph, instruction algebra, circuit collapse; historical
  lines 1–849) and
  `Layers.lean` the network/loss/training/inference suffix (850–1,648). Both keep
  `namespace Sanguine.NullTransformer` (declaration names unchanged); `Layers` imports `Core`; the
  umbrella `NullTransformer.lean` keeps the original module header and imports both. The lane may
  shift the boundary to a clean section start provided both files stay ≤1,500 and no name changes.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/NullTransformer/Core.lean`,
  `proof/corpus/Substrate/Algebra/Neural/NullTransformer/Layers.lean`,
  `proof/corpus/Substrate/Algebra/Neural/NullTransformer.lean`
- **Depends on:** `APEIRON-NEURAL-W1-T4`, `APEIRON-NEURAL-W1-T5`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.NullTransformer
  ```
- **Negative control:** In a disposable copy, mutate transformer forward evaluation; `lake build` must
  fail. Restore and verify exit 0.
- **Evidence to record:** Line counts of resulting files (all ≤1,500), compilation log.

**Result (closed; reference tree).** Split at historical line 849: `Core.lean` `5a0a14ef…` (716 lines),
`Layers.lean` `9d512074…` (828 lines, imports Core), umbrella `NullTransformer.lean` `49d0de71…`
(183 lines, historical header, imports both). Gate exit 0 (3261 jobs); control 0/1/0 (mutated `bl.wO`
-> `bl.wV` in `attention_is_two_circuits` rejected attributably). Statement changes: none. Provenance:
the historical file imports `Substrate.Algebra.Neural.RopeRelative`, which is outside the vendored
closure and absent from `vendor.sha256`; Core inlines `rdot`, `rope_score_is_relative` and
`rope_score_shift_invariant` from it. Two proof scripts were adjusted for the pinned Mathlib
(`differentiable_div`/`differentiable_div_const` in Core; `torus_point_is_recovered` in Layers). Under
the AI-track re-anchor the three files and the rest of the vendored tree are frozen reference material;
W2-T4 does not register names from the split composite and Wave 2 claims no exit.

#### APEIRON-NEURAL-W2-T2 · Rebase FrontierLadder

- **Scope:** Rebase `Substrate.Algebra.Neural.FrontierLadder` (267 lines), tracking the open seams
  over `CountLMFloor`.
- **Files:** `proof/corpus/Substrate/Algebra/Neural/FrontierLadder.lean`
- **Depends on:** `APEIRON-NEURAL-W1-T3`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.Neural.FrontierLadder
  ```
- **Negative control:** In a disposable copy, mutate a theorem statement in `FrontierLadder.lean`;
  `lake build` must fail. Restore and verify exit 0.
- **Evidence to record:** Compilation log, sha256 equality with historical file `cad186b0…`.

**Status.** Vendored by the integrator (`proof/corpus/Substrate/Algebra/Neural/FrontierLadder.lean`,
pristine digest `cad186b0d0c42d3242d51f65c91e7d9d008d37c9eb9f850c91a469e69aa8fbad`, manifest now 48
files); `lake build Substrate.Algebra.Neural.FrontierLadder` exits 0 with no adaptation. The task
reduces to the statement-mutation control and registration.

**Result (closed).** Gate exit 0 (3143 jobs). Control: baseline 0, red 1 — the `seam1_residual_bounded`
conclusion `≤` -> `<` is rejected at `213:2` through `CountLMFloor.refine_gap_le_weight_entropy` —
restore 0; live digest equals the historical digest. Headlines recorded: `reaches_conditioning_floor`,
`seam1_residual_bounded`, `parity_underdetermines_target`, `the_frontier_distance`,
`fold_preserves_correctness`, `estimation_gap_zero_at_truth`, `book_anchors_window`. The module's five
`Obligations` fields stay opaque and are not registered; `the_frontier_distance` remains conditional on
the unproven `hconst`.

#### APEIRON-NEURAL-W2-T3 · SpinorFold placement decision and rebase

- **Scope:** Subject to lead decision: vendor `Substrate/Algebra/NullCell/SpinorFold.lean` (303 lines)
  into `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean` and verify under `v4.31.0`.
- **Files:** `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean`
- **Depends on:** `APEIRON-NEURAL-W0-T2`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build Substrate.Algebra.NullCell.SpinorFold
  ```
- **Negative control:** In a disposable copy, mutate `nmul_assoc` statement; `lake build` must fail.
  Restore and verify exit 0.
- **Evidence to record:** Compilation log, sha256 equality with historical file `b5c274fa…`.

**Status.** Vendored by the integrator (`proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean`,
pristine digest `b5c274fab7dac63e41def5dde7facb6559b37c04773d0c8a80b287f8156e44e7`); placement
resolved to Wave 2 (`§7` 2). `lake build Substrate.Algebra.NullCell.SpinorFold` exits 0 with no
adaptation; the task reduces to the control and registration.

**Result (closed).** Gate exit 0 (1521 jobs). Control: baseline 0, red 1 — `nmul_assoc`'s conclusion
mutated to `nmul a (nmul c c)` is rejected at `174:78`, with the dependent `rotor_folds_into_weight`
red at `181:45` — restore 0; live digest equals the historical digest. The module declares 22
declarations in `namespace Sanguine.Spinor`; eight core mechanism names are registered in
`AxiomAudit.lean`.

#### APEIRON-NEURAL-W2-T4 · AxiomAudit registration for Wave 2

- **Scope:** Register Wave-2 headlines in `proof/corpus/AxiomAudit.lean` with `#audit` lines; verify
  axioms remain in `{propext, Classical.choice, Quot.sound}`.
- **Files:** `proof/corpus/AxiomAudit.lean`
- **Depends on:** `APEIRON-NEURAL-W2-T1`, `APEIRON-NEURAL-W2-T2`, `APEIRON-NEURAL-W2-T3`
- **Grade:** `proved`
- **Gate:**
  ```sh
  cd proof/corpus && lake build AxiomAudit
  ```
- **Negative control:** Seed unapproved axiom into audit list in disposable copy; audit fails.
  Restore and verify exit 0.
- **Evidence to record:** List of registered headlines, `#audit` output, build status.

**Status.** 15 of the wave-2 headlines (FrontierLadder, SpinorFold) are registered and
`lake build AxiomAudit` is green; NullTransformer names join when W2-T1 lands. Under the
[AI-track re-anchor](ai-track-reanchor.md) the vendored tree is reference material: no new
dependents or names, and the O(1) model is a separate epic with its own intent, spec and plan.

**Vendored tree milestone.** Seven adaptations integrated (`tools/neural-adapt/*.patch`:
CoordinateReadout, MadelungCell, MLPrimitivesBridge, RadixDescent, SparseAttentionBudget,
TrainingCollapse, UniverseSyntax); `adapted.sha256`
`b9a42df79b9728a7e7fbaeb6b8c0d3ed97abc1184b83e78bf8bedf510c20211a`; `bash tools/vendor-neural.sh`
applies all seven with pristine and adapted manifests verified (46 OK);
`lake build SubstrateVendored` exit 0 (3314 jobs), all 46 oleans present.

**W1-T7 status (green).** 18 wave-1 headlines plus the three new `GrowableRead` declarations
registered; the isolated probe resolves all names inside the closed cone. `lake build AxiomAudit` exit 0
(3877 jobs; `AxiomAudit.lean` `986448d0…`); bare `lake build` (full corpus) exit 0 (4003 jobs) with the
vendored tree at 46/46. Remaining for the wave exit: the W1-T3 control
(`refine_gap_zero_of_kappa_sufficient`, in flight) and the `Growable.Read` composition continuation
(`W5-T13-CONT8`, new module; draft at `/tmp/opencode/read-composition-draft.lean`).

**Wave-1 exit status (green).** All wave-1 rows pass on the same source identity: W0-T1/T2/T3 manifests
and roots; W1-T0 retired; W1-T0A–E adaptations (patches + `adapted.sha256` + vendor re-run 46 OK);
W1-T1–T6 module gates with statement-mutation controls; W1-T7 audit (3877 jobs); Wave-1 barrier
`lake build && lake build AxiomAudit` exit 0 (4003/3877 jobs). Open into wave 2: the NullTransformer
split shape and SpinorFold placement (`§7` decisions 2–3) and the FrontierLadder ledger; the Read
composition continuation is outside the neural scope.

### 6. Exit table

| Stage / Scope | Command | Target / Focus | Expected exit |
|---|---|---|---|
| W0 Manifest | `bash tools/vendor-neural.sh && sha256sum -c proof/corpus/Substrate/vendor.sha256` | 46 closure files | 0 |
| W0 Roots | `cd proof/corpus && lake build ApeironWordBridge` | Lake configuration | 0 |
| W0 Adapted manifest | `bash tools/vendor-neural.sh && sha256sum -c proof/corpus/Substrate/adapted.sha256` | Re-runnable adapted tree | 0 |
| W1 CollapseCore | `cd proof/corpus && lake build Substrate.CollapseCore` | Verified inapplicable (exit 0) | 0 |
| W1 UniverseSyntax | `cd proof/corpus && lake build Substrate.Algebra.Translation.UniverseSyntax` | Linter adaptation | 0 |
| W1 RadixDescent | `cd proof/corpus && lake build Closures.Engineering.RadixDescent` | Deprecation adaptation | 0 |
| W1 MadelungCell | `cd proof/corpus && lake build Substrate.Algebra.QuantumField.MadelungCell` | Proof repair | 0 |
| W1 CoordinateReadout | `cd proof/corpus && lake build Closures.Engineering.CoordinateReadout` | Linter adaptation | 0 |
| W1 SparseAttentionBudget | `cd proof/corpus && lake build Substrate.Algebra.Neural.SparseAttentionBudget` | Linter adaptation | 0 |
| W1 MirasCountLM | `cd proof/corpus && lake build Substrate.Algebra.Neural.MirasCountLM` | Focused module | 0 |
| W1 KneserNey | `cd proof/corpus && lake build Substrate.Algebra.Neural.KneserNey` | Focused module | 0 |
| W1 CountLMFloor | `cd proof/corpus && lake build Substrate.Algebra.Neural.CountLMFloor` | Focused module | 0 |
| W1 AttentionSpinor | `cd proof/corpus && lake build Substrate.Algebra.Neural.AttentionSpinor` | Focused module | 0 |
| W1 MLPrimitivesBridge | `cd proof/corpus && lake build Substrate.Algebra.Neural.MLPrimitivesBridge` | Focused module | 0 |
| W1 TrainingCollapse | `cd proof/corpus && lake build Substrate.Algebra.Neural.TrainingCollapse` | Focused module | 0 |
| W1 Audit | `cd proof/corpus && lake build AxiomAudit` | Wave-1 headlines | 0 |
| **Wave 1 Barrier** | `cd proof/corpus && lake build && lake build AxiomAudit` | Full corpus & audit | **0** |
| W2 NullTransformer | `cd proof/corpus && lake build Substrate.Algebra.Neural.NullTransformer` | Split composite | 0 |
| W2 FrontierLadder | `cd proof/corpus && lake build Substrate.Algebra.Neural.FrontierLadder` | Ledger module | 0 |
| W2 SpinorFold | `cd proof/corpus && lake build Substrate.Algebra.NullCell.SpinorFold` | Pending decision | 0 |
| W2 Audit | `cd proof/corpus && lake build AxiomAudit` | Wave-2 headlines | 0 |
| **Wave 2 Barrier** | `cd proof/corpus && lake build && lake build AxiomAudit` | Full corpus & audit | **0** |

### 7. Open decisions for the lead

1. **Root mechanism in `lakefile.lean` — resolved (`APEIRON-NEURAL-W0-T2`).** Option A adopted as a
   second `@[default_target] lean_lib SubstrateVendored` with
   `globs := #[.submodules `Substrate, .submodules `Closures]`. `ApeironWordBridge` keeps its
   explicit root list and does not import the vendored modules yet. Evidence: `lake build
   ApeironWordBridge` exit 0 (3821 jobs); `lake build SubstrateVendored` discovers all 46 modules
   with no `unknown target`; control 0/1/0. The pinned Lake accepts `globs` and `.submodules`
   recurses (scratch probe with nested `Vendor/B/C`).

2. **`SpinorFold` placement — resolved: Option A, vendor into stage 2 Wave 2** as
   `proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean` (303 lines). The module closes the
   NullCell family already vendored (`NullCell`, `NullCellCoding`); the fold-crossover link remains
   stage 4 work either way, and the vendored file makes the W2-T3 gate exercisable now.

3. **`NullTransformer` split decomposition shape — resolved: Option A, prefix/suffix at the network
   section boundary** (historical line 850; `Core.lean` = lines 1–849: the nonlinearity derivation, ops, node
   graph, instruction algebra, circuit collapse; `Layers.lean` = lines 850–1,648: Block/Net,
   shape/KV, loss, training, inference). Both files stay under 1,500 lines. Both declare
   `namespace Sanguine.NullTransformer`, so declaration names are unchanged; `Layers` imports `Core`;
   the umbrella `NullTransformer.lean` keeps the original module header and imports both. The lane
   may shift the boundary to a clean section start provided both files stay ≤1,500 and no name
   changes.

### 8. Barrier checklist and stop/rollback conditions

Run at wave barrier:
- [ ] Every task gate exit 0, with negative control executed and restored.
- [ ] No file owned by more than one task.
- [ ] File limit invariant: every vendored file ≤1,500 lines; every directory ≤12 immediate files.
- [ ] Axiom cone invariant: `#audit` verifies empty transitive axioms outside `{propext, Classical.choice, Quot.sound}`.
- [ ] No `sorry`, `admit` or `axiom` declarations in vendored tree.
- [ ] Pinned package `../sanguine-apeiron/proof` unmodified; manifest pin unchanged.

Stop and rollback conditions:
- If any admitted module encounters an unresolvable Mathlib semantic breaking change requiring theorem
  re-derivation, halt task, record minimal failing lemma, and escalate to lead.
- If three materially different adaptation attempts fail on a single module, stop and report the
  diagnostic blocker.
- Exceeding the 45-minute task budget or 10-minute command ceiling constitutes failure; do not retry
  without redesign.

### 9. Evidence block and path verification

All cited paths below resolve on the filesystem:

```text
/home/socol/Workspace/apeiron
docs/epics/neural-rebase.md
docs/epics/neural-rebase.md
docs/epics/neural-rebase.md
docs/epics/neural-rebase.md
docs/roadmap.md
docs/epics/agent-throughput-segmentation.md
docs/direction.md
docs/templates/epic.md
AGENTS.md
proof/corpus/lakefile.lean
proof/corpus/AxiomAudit.lean
proof/corpus/lean-toolchain
proof/corpus/lake-manifest.json
/home/socol/Workspace/sanguine
/home/socol/Workspace/sanguine/proof/lake-manifest.json
/home/socol/Workspace/sanguine/proof/lean-toolchain
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MirasCountLM.lean
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/KneserNey.lean
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/CountLMFloor.lean
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/AttentionSpinor.lean
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/TrainingCollapse.lean
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/NullTransformer.lean
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/FrontierLadder.lean
/home/socol/Workspace/sanguine/proof/Substrate/Algebra/NullCell/SpinorFold.lean
/home/socol/Workspace/sanguine/proof/Substrate/CollapseCore.lean
/home/socol/Workspace/spinor-ttl
/tmp/opencode/neural-closure.py
```

Path and whitespace gate:
```sh
cd /home/socol/Workspace/apeiron
git diff --no-index --check /dev/null docs/epics/neural-rebase.md
for p in $(sed -n '/^\/home\/socol\/Workspace/p;/^docs\//p;/^proof\/corpus/p;/^AGENTS.md/p;/^\/tmp\/opencode/p' docs/epics/neural-rebase.md); do
  test -e "$p" || echo "MISSING $p"
done
```

## Neural rebase inventory (APEIRON-NEURAL-INV)

- **Task:** APEIRON-NEURAL-INV — factual module inventory for the stage-2 Neural rebase spec
  (separate owner; spec still to be written).
- **Grade:** `anchor` — inventory only. No theorem is checked, no Lean or cargo build is run, no
  `proof/regen.sh`. Counts and imports are command output; historical grades are retained, not
  re-earned.
- **Scope:** one row per in-scope module (paths, sha256, line count, declaration counts,
  `sorry`/`admit`/`axiom` scan, cross-module imports, paper-INDEX headline names), the old identity
  pin, a candidate first wave, and per-module rebase risks. It changes no source, theorem or gate.
- **Source root for the table paths:** `/home/socol/Workspace/sanguine` (historical checkout).

### Commands used

```sh
cd /home/socol/Workspace/sanguine
sha256sum <9 module paths>            # digests, table column 2
wc -l <9 module paths>                # line counts, table column 3
rg -c 'theorem|lemma' <file>          # thm+lem column (matching lines)
rg -c '\b(def|structure|instance)\b' <file>
rg -c 'sorry|admit|axiom' <file>      # scan column
rg -n 'sorry|admit|axiom' <file> | head -3
rg -n '^import ' <file>               # import lines
rg -n '^\s*(theorem|lemma)\b' <file>  # anchored declaration counts (footnote)
rg -n '^\s*(def|structure|instance|abbrev|noncomputable def)\b' <file>
cat -n /home/socol/Workspace/sanguine/lean-toolchain
cat -n /home/socol/Workspace/sanguine/proof/lean-toolchain
cat -n /home/socol/Workspace/sanguine/proof/lake-manifest.json
rg -n <INDEX theorem names> <9 module paths>          # name-level INDEX match
rg -n <Substrate module> /home/socol/Workspace/sanguine/proof/lakefile.lean
test -e <each cited path>                              # path gate, see §6
```

The handoff named a manifest at the sanguine repository root; `test -e` reports no file there. The
manifest that exists and is quoted is at `/home/socol/Workspace/sanguine/proof/lake-manifest.json`.

### 1. Module table

One row per module. Paths are relative to `/home/socol/Workspace/sanguine` (the resolved absolute
forms are listed in §6); every row resolves with `test -e`. `thm+lem` and `def+struct+inst` are
`rg -c` matching-line counts; anchored declaration counts are in the footnote. The
`sorry|admit|axiom` column is the substring scan — every hit is inside prose (`admits`, or a
sentence that says a cite is `sorry`/`axiom`-free), and an anchored
`^\s*(sorry|admit|axiom)\b` scan returns nothing. "INDEX headlines" are the theorem names the
paper1/paper2 `INDEX.md` files name that are also declared in this file (name-level match; the INDEX
files name release-bundle paths, not Neural paths).

| module | sha256 | lines | thm+lem | def+struct+inst | sorry\|admit\|axiom (count; first 3 lines) | Substrate / cross-module imports | INDEX headlines declared here |
|---|---|---|---|---|---|---|---|
| `proof/Substrate/Algebra/Neural/MirasCountLM.lean` | `9738ee17f7f7d130f69ab6c345743e7c2c06517d624a27d0cb4c7e6dd359577e` | 145 | 9 | 3 | 0; — | `Substrate.Algebra.NullCell.NullCellCoding` | none |
| `proof/Substrate/Algebra/Neural/AttentionSpinor.lean` | `7cb47e7df62edc4bfb95ed4b20016563ee53537d92d388e39f8752c13b13935b` | 713 | 33 | 22 | 0; — | `Substrate.StationaryPhase.SpinorLoop` | none |
| `proof/Substrate/Algebra/Neural/NullTransformer.lean` | `636ba26a1fc3c67d25e50d298f977eb2546931dde2ea02d6d4d31d522e472fa0` | 1648 | 95 | 46 | 2; 165, 349 | `Substrate.Algebra.NullCell.NullCell`; `Substrate.Algebra.Neural.AttentionSpinor`; `Substrate.Algebra.Neural.RopeRelative` | none |
| `proof/Substrate/Algebra/Neural/FrontierLadder.lean` | `cad186b0d0c42d3242d51f65c91e7d9d008d37c9eb9f850c91a469e69aa8fbad` | 267 | 24 | 4 | 2; 25, 77 | `Substrate.Algebra.Neural.CountLMFloor` | none |
| `proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean` | `32bba81f7b57dd38db1a23b475195bdeaaf8430c3a9402ba49f452ec012e50c0` | 452 | 26 | 9 | 1; 341 | `Substrate.Algebra.Neural.TrainingCollapse`; `Substrate.Algebra.Security.HashTrinityCore`; `Substrate.Algebra.NullCell.Opportunity`; `Substrate.Collapse` | `osFuse_is_add`, `osFuse_grouping_free`, `output_error_is_function_distortion`, `aware_scale_is_the_chart`, `weight_only_is_dual_blind` |
| `proof/Substrate/Algebra/Neural/CountLMFloor.lean` | `d82351a7d4dd167d79376abcc3fb5e43118ebe57c9a44fc0a1f2b3279d3c4fb9` | 537 | 29 | 11 | 0; — | `Substrate.Algebra.Neural.SparseAttentionBudget` | none |
| `proof/Substrate/Algebra/Neural/KneserNey.lean` | `4611c4071d735a8e3e6e6246da45257096b74f256fdd32e06dbb561ccdb6e768` | 172 | 9 | 9 | 0; — | none (Mathlib only) | none |
| `proof/Substrate/Algebra/Neural/TrainingCollapse.lean` | `9eeb8663fcd765d7e73e1ce932cbd43ce979ccedb05095dc3ecb3247704d1c33` | 830 | 48 | 17 | 1; 534 | `Substrate.Algebra.QuantumField.QuantumClassicalIsomorphism`; `Substrate.Collapse` | `ridgeHead_unique`, `gram_stack` |
| `proof/Substrate/Algebra/NullCell/SpinorFold.lean` | `b5c274fab7dac63e41def5dde7facb6559b37c04773d0c8a80b287f8156e44e7` | 303 | 24 | 11 | 0; — | `Substrate.Algebra.NullCell.NullCell` | none |

Footnote: anchored declaration counts (`^\s*(theorem|lemma)\b` / `^\s*(def|structure|instance|abbrev|noncomputable def)\b`)
are MirasCountLM 6/4, AttentionSpinor 31/17, NullTransformer 84/48, FrontierLadder 7/1,
MLPrimitivesBridge 20/6, CountLMFloor 26/11, KneserNey 6/9, TrainingCollapse 34/15, SpinorFold 19/8.

### 2. Old identity block

Quoted with file and line.

- `/home/socol/Workspace/sanguine/lean-toolchain:1`:
  `leanprover/lean4:v4.30.0-rc2`
- `/home/socol/Workspace/sanguine/proof/lean-toolchain:1`:
  `leanprover/lean4:v4.30.0-rc2` (same string)
- `/home/socol/Workspace/sanguine/proof/lake-manifest.json:8`:
  `   "rev": "89d44072ff8960893f469c67f75280db4fe4023a",`
  with `:9` `   "name": "mathlib",` and `:11` `   "inputRev": "master",`.
  Manifest header `:1` `{"version": "1.2.0",`, package name `:94` ` "name": "«sanguine-proof»",`.

Current pin for contrast (from `/home/socol/Workspace/apeiron/proof/corpus`):

- `/home/socol/Workspace/apeiron/proof/corpus/lean-toolchain:1`:
  `leanprover/lean4:v4.31.0`
- `/home/socol/Workspace/apeiron/proof/corpus/lake-manifest.json:8`:
  `   "rev": "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f",` (`:9` `"name": "mathlib",`)

A manifest at the sanguine repository root does not exist; the digest-bearing manifest is
`/home/socol/Workspace/sanguine/proof/lake-manifest.json`.

### 3. Metadata sources: paper INDEX and spinor-ttl

**Theorem-name lists.** The paper1 INDEX
(`/home/socol/Workspace/sanguine/release/paper1/INDEX.md`) names, by release file:
`/home/socol/Workspace/sanguine/release/paper1/Numerics.lean` → `accumulator_fits`,
`lossy_error_is_additive`, `token_survives_margin`, `numerics_seam`,
`output_error_is_function_distortion`, `aware_scale_is_the_chart`, `weight_only_is_dual_blind`,
`osFuse_is_add`, `osFuse_grouping_free`, `flash_attention_exact`;
`/home/socol/Workspace/sanguine/release/paper1/Train.lean` → `ridgeHead_unique`, `gram_stack`,
`streaming_law`; `/home/socol/Workspace/sanguine/release/paper1/Cell.lean` →
`null_cell_characterization`; `/home/socol/Workspace/sanguine/release/paper1/Spinor.lean` →
`spinor_subtrinity`; `/home/socol/Workspace/sanguine/release/paper1/Ops.lean` →
`six_ops_one_cell`; `/home/socol/Workspace/sanguine/release/paper1/Seed.lean` →
`rexpand_rcollapse`, `Gen.decode_encode`. The paper2 INDEX
(`/home/socol/Workspace/sanguine/release/paper2/INDEX.md`) names
`meaning_is_the_only_freedom`, `fromTerm_unique`, `state_is_a_fromTerm`, `lift_is_format_agnostic`,
`the_one_object`, `constantStep_fixed_point`, `affine_orbit_closed_form`,
`fixed_point_is_bulk_ratio`, `loops_are_closed_forms`, `canonical_floor_unique`, `confluence`,
`evalIO_syscall`, `capability_orthogonal_awareness`, `no_self_reference`, declared over
`/home/socol/Workspace/sanguine/release/paper2/Object.lean` and
`/home/socol/Workspace/sanguine/release/paper2/Loop.lean`; the INDEX marks its `Machine`, `Syscall`
and `Servitor` rows in progress and no release file for them exists.

Of these, only `ridgeHead_unique`, `gram_stack` (declared in the TrainingCollapse module) and
`osFuse_is_add`, `osFuse_grouping_free`, `output_error_is_function_distortion`,
`aware_scale_is_the_chart`, `weight_only_is_dual_blind` (declared in the MLPrimitivesBridge module)
occur in the in-scope modules. No in-scope module path is named by either INDEX.

**spinor-ttl measured claims and their mapped modules**
(`/home/socol/Workspace/spinor-ttl/PAPER.md`, `/home/socol/Workspace/spinor-ttl/STATUS.md`):

| measured claim (source) | mapped module (as stated) |
|---|---|
| count-LM 2.205 b/B in 1.7 s vs 2-layer byte transformer 2.392 b/B in 20.9 s; Lean ref 2.239 vs 2.469 (PAPER §2, `/home/socol/Workspace/spinor-ttl/bench/beats_transformer.py`) | `/home/socol/Workspace/spinor-ttl/src/spinor_ttl/countlm.py`; counts-cited Lean: KneserNey, CountLMFloor, MirasCountLM |
| `osFuse_is_add` / `osFuse_grouping_free` (PAPER §1, names `SpinorFold.lean`) | **drift:** the theorems are declared at `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:174,182`, not in `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/NullCell/SpinorFold.lean` |
| `output_error_is_function_distortion`, `aware_scale_is_the_chart`, `weight_only_is_dual_blind` (PAPER §3, MLPrimitivesBridge §6) | `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:304,333,344` (matches) |
| `kn_floor_refinement` (PAPER §4, MLPrimitivesBridge §6b) | `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:385` (matches) |
| seal 4.2665 bits/token @4.45 b/w vs NF4 4.3032 @4.50; vs Q4_K_M 4.2121/4.2191 @6.34 (PAPER §3) | `/home/socol/Workspace/spinor-ttl/src/spinor_ttl/seal.py`; mechanism: MLPrimitivesBridge §6 |
| finetune fold⊕base 0.724 b/B vs LoRA 0.850, fold alone 0.957, frozen base 0.910; exact unlearning (PAPER §4) | `/home/socol/Workspace/spinor-ttl/src/spinor_ttl/finetune.py` (`FoldAdapter`); mechanism: MLPrimitivesBridge `kn_floor_refinement`, MirasCountLM `forget_hebb` |
| ridge fold 16–71× vs full-batch Adam at the same optimum (STATUS, `/home/socol/Workspace/spinor-ttl/bench/train_fold_vs_grad.py`) | `/home/socol/Workspace/spinor-ttl/bench/test_sparse_ridge_fold.py`; mechanism: TrainingCollapse `ridgeHead`, `/home/socol/Workspace/sanguine/proof/Runtime/Model.lean` `BU.foldRow` |

### 4. Candidate first wave

Load-bearing for active test-time learning / lower-cost inference, per the paper INDEX and
spinor-ttl. One rationale line each.

| module | rationale |
|---|---|
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MirasCountLM.lean` | Test-time memory as the exact fold: `hebb`/`forget` are the order-free update and its exact inverse (`hebb_comm`, `forget_hebb`); the `hebb` rule is the count-table primitive. |
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/KneserNey.lean` | The smoothing lever that closes the estimation gap (`kn_normalizes`); spinor-ttl counts on it for both count-LM and finetune. Self-contained (Mathlib only). |
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/CountLMFloor.lean` | The bits/byte decomposition and the two levers (KL→0, conditioning→full) that name the lower-cost inference target; declares the open `kappa_sufficient` seam. |
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/AttentionSpinor.lean` | Attention as an `A=I` spinor scan and exact online-softmax carry: `O(1)` state per step, the lower-cost inference mechanism. |
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean` | The op-level bridge carrying the flash-attention chart (`osFuse_*`) and the quant-dual law spun out in spinor-ttl §3; also `kn_floor_refinement`. |
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/TrainingCollapse.lean` | Closed-form ridge solve and exact streaming/removal fold (`ridgeHead_unique`, `gram_stack`, `normal_equations_*`): the spinor-ttl "training fold" speed win. |
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/NullCell/SpinorFold.lean` (decision item) | The fold/softmax algebra spinor-ttl cites for the streaming merge; carries the real `spinorFold`/`nmul_assoc` content, but the cited `osFuse_*` names live elsewhere (drift above). The intent leaves "stage 2 or fold crossover" open — owner decision. |

Clearly downstream, can wait:

| module | why it can wait |
|---|---|
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/FrontierLadder.lean` | A prose ledger over CountLMFloor with only four native facts; every rung cites a theorem in another module. It cannot be re-bound before the modules it names. |
| `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/NullTransformer.lean` | A composite consumer of AttentionSpinor, RopeRelative and the MLPrimitivesBridge/TrainingCollapse algebra; 1648 lines and the largest Mathlib surface. It re-states nothing those modules do not already carry and should follow them. |

The NullCell world/terrain modules and the physics, security and game rows are excluded by the
intent's non-goals and are not in this inventory.

### 5. Rebase risks per module

"Re-homing" means an import under the old `Substrate.*` / `Closures.*` roots that must be carried
into the rebase home (`proof/corpus/Neural/` importing the pinned package, per the intent). Every
in-scope Mathlib import path and every Mathlib declaration named in the docstrings below was checked
**by file/name grep against the current Mathlib tree**
(`/home/socol/Workspace/apeiron/proof/corpus/.lake/packages/mathlib/Mathlib`); no build was run.

- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MirasCountLM.lean` — re-home
  `Substrate.Algebra.NullCell.NullCellCoding`
  (`/home/socol/Workspace/sanguine/proof/Substrate/Algebra/NullCell/NullCellCoding.lean`, itself
  importing `NullCell`). Mathlib: `Mathlib.Tactic` present.
- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/AttentionSpinor.lean` — re-home
  `Substrate.StationaryPhase.SpinorLoop`
  (`/home/socol/Workspace/sanguine/proof/Substrate/StationaryPhase/SpinorLoop.lean`), whose own
  imports pull `AffineLoopCollapse`, `Ski.PolyIotaTermAlgebra` and 8 Mathlib modules; the widest
  out-of-directory fan-in of the wave. Mathlib: `Data.Matrix.Mul`,
  `SpecialFunctions.Trigonometric.Basic` present.
- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/NullTransformer.lean` — re-home
  `NullCell.NullCell` plus the two in-directory imports `Neural.AttentionSpinor`,
  `Neural.RopeRelative`. Mathlib: `SpecialFunctions.Sqrt`, `SpecialFunctions.Log.Deriv`,
  `Calculus.Deriv.Slope`, `Calculus.FDeriv.Pi` present.
- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/FrontierLadder.lean` — re-home
  `Neural.CountLMFloor` only; imports no Mathlib module directly.
- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean` —
  highest external fan-in: re-home `Security.HashTrinityCore` (Mathlib-only),
  `NullCell.Opportunity` (pulling `Trinity.NonBeing`, `StationaryPhase.BalancedSingularity`,
  `Substrate.Collapse`), and `Substrate.Collapse` (pulling `Substrate.CollapseCore`), plus the
  in-directory `Neural.TrainingCollapse`. Mathlib: `Data.List.Sort` present; the sort row rides
  `List.mergeSort_perm` and `List.Perm.eq_of_pairwise'`, both present.
- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/CountLMFloor.lean` — re-home
  `Neural.SparseAttentionBudget`
  (`/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/SparseAttentionBudget.lean`),
  which itself imports `Substrate.Algebra.Bulk.BulkExhaustiveness`
  (`/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Bulk/BulkExhaustiveness.lean`) **and**
  `Closures.Engineering.CoordinateReadout`
  (`/home/socol/Workspace/sanguine/proof/Closures/Engineering/CoordinateReadout.lean`): a second
  root namespace (`Closures`, not `Substrate`) that a rebase must carry. Mathlib:
  `Algebra.BigOperators.Fin`, `Algebra.Order.BigOperators.Ring.Finset`,
  `SpecialFunctions.Log.Basic`, `SpecialFunctions.Log.NegMulLog`, `Data.Real.Basic`, the tactic
  modules present. The cited names `Real.log_le_sub_one_of_pos`,
  `Real.strictConcaveOn_negMulLog`, `Finset.sum_mul_sq_le_sq_mul_sq` are present.
- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/KneserNey.lean` — no
  cross-module import; Mathlib-only (`Mathlib.Tactic`). Lowest re-homing risk.
- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/Neural/TrainingCollapse.lean` — re-home
  `QuantumField.QuantumClassicalIsomorphism`
  (`/home/socol/Workspace/sanguine/proof/Substrate/Algebra/QuantumField/QuantumClassicalIsomorphism.lean`,
  pulling `QuantumField.MadelungCell`, `QuantumField.WickRotation`, `NullCell.NullCell`) and
  `Substrate.Collapse`. Mathlib surface largest of the wave; all 18 imported module paths and the
  cited names `Matrix.posSemidef_conjTranspose_mul_self`, `Matrix.PosDef.smul`,
  `Matrix.PosDef.posSemidef_add`, `Matrix.PosDef.isUnit`
  (`/home/socol/Workspace/apeiron/proof/corpus/.lake/packages/mathlib/Mathlib/LinearAlgebra/Matrix/PosDef.lean:507`),
  `Matrix.posSemidef_diagonal_iff`, `PosSemidef.conjTranspose_mul_mul_same`, `Complex.exp_conj`,
  `Matrix.mul_apply` are present.
- `/home/socol/Workspace/sanguine/proof/Substrate/Algebra/NullCell/SpinorFold.lean` — re-home
  `NullCell.NullCell` (`/home/socol/Workspace/sanguine/proof/Substrate/Algebra/NullCell/NullCell.lean`)
  only; Mathlib-free. Low risk.

Toolchain-version features: a `rg -n '^import '` and
`rg -n '^\s*(def|structure|instance|abbrev|noncomputable def|theorem|lemma)\b'` pass over the nine
files found no syntax beyond `noncomputable section`, `abbrev`, `@[simp]`, `variable`, `open` and
standard tactic blocks. The old pin `v4.30.0-rc2` and the current `v4.31.0` are adjacent Lean
releases; the risk found is the cross-module re-homing above, not a Lean-syntax feature.
`sorry`/`admit`/`axiom` do not occur as declarations, so the intent's kernel discipline is not
already violated by these files.

Unverified (not a build): whether each module compiles at `v4.31.0`, and whether a declaration
renamed inside Mathlib while its module path stayed the same. Both require the focused module gate
the spec will run; this inventory establishes only path and name existence.

### 6. Evidence block

Commands and output summary (all run under a 30-minute budget; no command approached it).

```text
sha256sum <9 paths>
  9738ee17f7f7d130f69ab6c345743e7c2c06517d624a27d0cb4c7e6dd359577e  proof/Substrate/Algebra/Neural/MirasCountLM.lean
  7cb47e7df62edc4bfb95ed4b20016563ee53537d92d388e39f8752c13b13935b  proof/Substrate/Algebra/Neural/AttentionSpinor.lean
  636ba26a1fc3c67d25e50d298f977eb2546931dde2ea02d6d4d31d522e472fa0  proof/Substrate/Algebra/Neural/NullTransformer.lean
  cad186b0d0c42d3242d51f65c91e7d9d008d37c9eb9f850c91a469e69aa8fbad  proof/Substrate/Algebra/Neural/FrontierLadder.lean
  32bba81f7b57dd38db1a23b475195bdeaaf8430c3a9402ba49f452ec012e50c0  proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean
  d82351a7d4dd167d79376abcc3fb5e43118ebe57c9a44fc0a1f2b3279d3c4fb9  proof/Substrate/Algebra/Neural/CountLMFloor.lean
  4611c4071d735a8e3e6e6246da45257096b74f256fdd32e06dbb561ccdb6e768  proof/Substrate/Algebra/Neural/KneserNey.lean
  9eeb8663fcd765d7e73e1ce932cbd43ce979ccedb05095dc3ecb3247704d1c33  proof/Substrate/Algebra/Neural/TrainingCollapse.lean
  b5c274fab7dac63e41def5dde7facb6559b37c04773d0c8a80b287f8156e44e7  proof/Substrate/Algebra/NullCell/SpinorFold.lean

wc -l <9 paths>   -> 145, 713, 1648, 267, 452, 537, 172, 830, 303 (total 5067)

rg -c 'sorry|admit|axiom' <9 paths>  -> 0,0,2,2,1,0,0,1,0 (all hits prose; anchored scan empty)
  NullTransformer.lean:165, :349; FrontierLadder.lean:25, :77; MLPrimitivesBridge.lean:341;
  TrainingCollapse.lean:534

rg -n '^\s*(sorry|admit|axiom)\b' <9 paths>  -> (none)

rg -n '^import ' <9 paths>  -> the import lines transcribed in the table

cat -n /home/socol/Workspace/sanguine/lean-toolchain                 -> 1  leanprover/lean4:v4.30.0-rc2
cat -n /home/socol/Workspace/sanguine/proof/lean-toolchain           -> 1  leanprover/lean4:v4.30.0-rc2
cat -n /home/socol/Workspace/sanguine/proof/lake-manifest.json       -> :8  "rev": "89d44072ff8960893f469c67f75280db4fe4023a",
                                                                        :9  "name": "mathlib",

cat -n /home/socol/Workspace/apeiron/proof/corpus/lean-toolchain          -> 1  leanprover/lean4:v4.31.0
rg -n '"name": "mathlib"|"rev"' /home/socol/Workspace/apeiron/proof/corpus/lake-manifest.json
                                    -> :8 "rev": "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f", :9 "name": "mathlib"

rg -n <INDEX theorem names> <9 paths>
  -> MLPrimitivesBridge.lean:174,182,304,333,344; TrainingCollapse.lean:223,245
  -> No in-scope path is named by either INDEX.

rg -n <Mathlib module names> /home/socol/Workspace/apeiron/proof/corpus/.lake/packages/mathlib/Mathlib
  -> all imported Mathlib module paths present; all cited declaration names present.

git -C /home/socol/Workspace/apeiron/proof/corpus/.lake/packages/mathlib rev-parse HEAD
  -> fabf563a7c95a166b8d7b6efca11c8b4dc9d911f
git -C /home/socol/Workspace/sanguine/proof/.lake/packages/mathlib rev-parse HEAD
  -> 89d44072ff8960893f469c67f75280db4fe4023a
```

Path gate (table paths resolved against `/home/socol/Workspace/sanguine`; every other cited path is
absolute) and the whitespace gate:

```sh
cd /home/socol/Workspace/apeiron
git diff --no-index --check /dev/null docs/epics/neural-rebase.md
for p in <resolved path list, §6 evidence>; do test -e "$p" || echo "MISSING $p"; done
```

Both are logged verbatim at `/tmp/opencode/neural-inv-gate.log`; the required result is no
whitespace diagnostic and no `MISSING` line.
