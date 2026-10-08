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
- Re-running the spinor-ttl measurements (stage 3 re-runs them).
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


## Stage 3 — learning track (intent, spec, plan)

### Intent — Learning track (verified-AI sequence, stage 3)

- **Owner:** Apeiron lead
- **Source revision:** `024ec32`
- **Status:** draft
- **Spec:** [`docs/epics/neural-rebase.md`](neural-rebase.md) (draft)
- **Plan:** [`docs/epics/neural-rebase.md`](neural-rebase.md) (draft)

```text
SCOPE: docs/epics/neural-rebase.md only; planning, intent, spec and plan for learning track
REGISTER: learning track refinement and implementation (docs/direction.md#programme-outcomes,
          "Active test-time learning and lower-cost useful AI"); excludes execution and grade changes
CORPUS: vendored neural statements proof/corpus/Substrate/Algebra/Neural/{MirasCountLM, KneserNey,
        CountLMFloor, AttentionSpinor, MLPrimitivesBridge, TrainingCollapse}.lean and
        proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean; canonical definitions
        Apeiron.Table.{HMap, get, insertWith, build} and Apeiron.Trichotomy.AnyRing
REFERENCE: ../spinor-ttl (a05bb1b); ../fold (d497161f, e037ba11 modelfold/);
           original commits f875705d, eaeca005, 3c03ff38, db5397b2, d79460e8, 83d9cc8a;
           docs/plan.md Learning track
GATE: python3 tools/check-claims.py
GRADE: anchor/planning
```

#### Problem

The formal neural foundations and historical statements were vendored into `proof/corpus/Substrate/Algebra/Neural/` and audited under Lean `v4.31.0` in stage 2 (`APEIRON-NEURAL-W1`). However, these Lean theorems currently stand as standalone formal statements without executing Rust implementations in this repository. The empirical winners for training, finetuning, and quantization exist only in external checkouts (`../spinor-ttl` at HEAD `a05bb1b` and `../fold` at HEAD `d497161f`). Without implementing and verifying the executing carriers in this crate, the AI-track outcome ("Active test-time learning and lower-cost useful AI") lacks primary artifact backing, leaving open obligations between the formal theorems and executing binaries.

#### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Retain `../spinor-ttl` Python/PyTorch modules as production | Violates primary artifact pillar (AGENTS.md §4 obligation 1): external Python runtimes with unverified dependencies cannot be extracted or linked to the Lean kernel. |
| Rely on `../fold/modelfold` via runtime linkage | Cross-repo runtime linkage introduces circularity and unverified surface; apeiron requires in-tree verified primitives. |
| Re-derive mechanisms from first principles without subsumption | Violates AGENTS.md §3a ("Subsume the empirical best before deriving"): re-deriving discards measured empirical advantages and verified baselines. |
| Treat stage-2 standalone Lean proofs as DoD closure | Standalone theorems without an executing Rust correspondence do not establish end-to-end verified learning (AGENTS.md §4 obligation 3). |
| Optimize training speed before verifying bits/token | Violates programme direction (owner 2026-10-08): primary criterion is held-out bits/token at a stated bits/weight; speed is secondary. |

#### Completion condition

All six learning track mechanisms ((a) seal quantizer, (b) FoldAdapter, (c) ridge fold, (d) KN count-LM, (e) gptq / 4→3-trit planes, (f) foldnet) are implemented under `src/learn/` (≤3 files per task), with passing focused test suites (`cargo test --release --locked --test learn_<x>`), explicit negative mutation controls, and formal Lean refinement links linking extracted Rust structures to `proof/corpus/Substrate/Algebra/Neural/`. Execution passes without inference speed regression on the pinned golden model (Qwen2.5-0.5B SEQHASH `9092dbbaff119a2e`).

#### Invariants

1. **No inference speed regression:** `tg128` and `pp512` token rates on Qwen2.5-0.5B Q4_K_M do not regress against the speed-guard baseline on the same hardware.
2. **Metric precedence:** held-out bits/token (or bits/byte) at stated bits/weight is the primary acceptance criterion. Speedups are secondary.
3. **Axiom cone:** refinements audit within `{propext, Classical.choice, Quot.sound}` without `sorry`/`admit`/`axiom`.
4. **Exact/float factoring:** exact integer and rational arithmetic is extracted via Aeneas and formally refined; floating-point evaluation is tested against analytical references.
5. **Task decomposition:** each implementation task touches ≤3 files, declares one focused gate and one negative control, and executes in isolated worktree lanes.

#### Verification hypotheses

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| Seal activation-calibrated quantization reaches bits/token parity | tested | `cargo test --release --locked --test learn_seal` |
| Seal output distortion minimizes function distortion $\sum \tau_j r_j^2$ | proved | `MLPrimitivesBridge.output_error_is_function_distortion` (`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:304`) |
| FoldAdapter forward-only finetuning achieves target b/B | tested | `cargo test --release --locked --test learn_foldadapt` |
| FoldAdapter additive merge and unlearning commute | proved | `MLPrimitivesBridge.osFuse_is_add` (`proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:174`), `osFuse_grouping_free` (`:182`), `MirasCountLM.forget_hebb` (`proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:101`) |
| Ridge fold streaming Gram accumulation and SPD solve reach optimum | tested | `cargo test --release --locked --test learn_ridge` |
| Ridge normal equations yield unique optimum | proved | `TrainingCollapse.ridgeHead_unique` (`proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean:223`), `gram_stack` (`:245`) |
| KN count-LM achieves $\le 2.205$ b/B | tested | `cargo test --release --locked --test learn_count` |
| Kneser–Ney distribution normalizes | proved | `KneserNey.kn_normalizes` (`proof/corpus/Substrate/Algebra/Neural/KneserNey.lean:98`), `MirasCountLM.countlm_is_exact_miras` (`proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:138`) |
| GPTQ error-feedback produces lower output distortion than naive rounding | tested | `cargo test --release --locked --test learn_gptq` |
| FoldNet nonlinear random feature expansion solves non-linear benchmarks | tested | `cargo test --release --locked --test learn_foldnet` |

#### Non-goals

- Large-scale distributed pre-training runs (deferred to stage 6).
- New neural architectural searches or novel optimizer derivations.
- Modifying the pinned canonical corpus at `../sanguine-apeiron/proof`.
- Modifying the frozen stage-2 vendored Lean files in `proof/corpus/Substrate/`.

#### Trust boundary

The learning track claims cover in-tree Rust implementations in `src/learn/`, their unit and integration tests under `tests/`, their Aeneas extraction models, and Lean kernel-checked refinement links to the vendored Neural modules. They do not cover third-party host BLAS libraries, CUDA driver execution paths, external dataset tokenization pipelines, or LLVM target-specific codegen quirks.

#### Open questions

- **GGUF write target:** whether the seal quantizer GGUF export writes custom tensor types or standard Q4_K / Q8_0 blocks with sidecar scale tables.
- **Counting face backing:** whether `CountLM` and `FoldAdapter` consume the refined `FreqMap` directly or utilize table-specialized storage arrays.

---

### Specification — Learning track (verified-AI sequence, stage 3)

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/neural-rebase.md`](neural-rebase.md)
- **Status:** draft

```text
SCOPE: docs/epics/neural-rebase.md only; specification for learning track mechanisms
REGISTER: learning track mechanism specification and refinement links
CORPUS: proof/corpus/Substrate/Algebra/Neural/{MirasCountLM, KneserNey, CountLMFloor,
        AttentionSpinor, MLPrimitivesBridge, TrainingCollapse}.lean;
        proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean
REFERENCE: ../spinor-ttl; ../fold/modelfold; commits f875705d, eaeca005, 3c03ff38, db5397b2,
           d79460e8, 83d9cc8a
GATE: python3 tools/check-claims.py
GRADE: anchor/planning
```

#### Scope and admissible domain

This specification covers six learning, finetuning, and quantization mechanisms reimplemented in Rust in this crate, with formal refinement to vendored Lean declarations in `proof/corpus/Substrate/Algebra/Neural/`.

| Mechanism | Subsumed Base | Lean Reference | Primary Metric |
|---|---|---|---|
| (a) Seal quantizer | `../spinor-ttl/src/spinor_ttl/seal.py`, `../fold/modelfold/src/quant.rs` | `MLPrimitivesBridge.output_error_is_function_distortion` | bits/token @ stated bits/weight |
| (b) FoldAdapter | `../spinor-ttl/src/spinor_ttl/finetune.py`, `../fold/modelfold/src/train.rs` | `MLPrimitivesBridge.osFuse_is_add`, `kn_floor_refinement` | bits/byte (b/B) on domain adaptation |
| (c) Ridge fold | `../spinor-ttl/src/spinor_ttl/foldtrain.py`, `../fold/modelfold/src/foldtrain.rs` | `TrainingCollapse.ridgeHead_unique`, `gram_stack` | wall-clock time to Adam optimum |
| (d) KN count-LM | `../spinor-ttl/src/spinor_ttl/countlm.py`, `../fold/modelfold/src/countlm.rs` | `KneserNey.kn_normalizes`, `MirasCountLM.countlm_is_exact_miras` | bits/byte (b/B) on byte prediction |
| (e) GPTQ 4→3-trit | `../fold/modelfold/src/quant.rs`, `../fold` `d79460e8:hashtrinity/src/gptq.rs` | `MLPrimitivesBridge.output_error_is_function_distortion` (metric) | output distortion $\sum \tau_j r_j^2$ |
| (f) FoldNet | `../fold/modelfold/src/arch.rs`, `../fold` `83d9cc8a:hashtrinity/src/foldnet.rs` | `TrainingCollapse.ridgeHead_unique`, `gram_stack` | classification accuracy on XOR / multimodal |

#### Representation relation

| Layer | Symbol | Representation |
|---|---|---|
| Rust carrier (exact) | `Cell<i64>`, integer counts, bit-packed trits | Discrete coordinate addressing, exact integer monoids $(\mathbb{Z}, +)$ |
| Rust carrier (float) | `f32`, `f64`, vector buffers | Floating-point arrays evaluated against golden analytical references |
| Lean object | Real matrix `Matrix (Fin n) (Fin d) ℝ`, `Memory K`, `pKN` | Continuous formal structures in Mathlib and `proof/corpus/Substrate/` |
| Exact/Float boundary | Quantization grid, Gram accumulator, count updates | Integer kernels extracted via Aeneas; solve and loss evaluation in float |

#### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `seal_calibrated_distortion` | At equal bits/weight, the seal grid's calibrated distortion $\sum \tau_j r_j^2$ is no larger than round-to-nearest's on the same calibration set | tested | `cargo test --release --locked --test learn_seal` |
| `foldadapter_unlearning` | Exact unlearning holds: $\text{fold}(A + B) - \text{fold}(B) = \text{fold}(A)$ bit-for-bit | tested + proved | `tests/learn_foldadapt.rs`; `MirasCountLM.forget_hebb` |
| `ridge_unique_optimum` | Regularized drag $U = X^T X + \lambda I$ is strictly positive definite for $\lambda > 0$ | proved | `TrainingCollapse.ridgeHead_unique` |
| `countlm_normalization` | KN backoff distribution sums to 1: $\sum_w p_{KN}(w \mid \text{ctx}) = 1$ | proved | `KneserNey.kn_normalizes` |
| `gptq_error_feedback` | Column error-feedback achieves lower output distortion than naive rounding | tested | `tests/learn_gptq.rs` |
| `foldnet_nonlinearity` | Fixed random nonlinear expansion separates XOR benchmark (>99% accuracy) | tested | `tests/learn_foldnet.rs` |

#### Failure semantics

| Condition | Failure mode | Diagnostic |
|---|---|---|
| Non-positive regularization $\lambda \le 0$ | Refusal at solve entry | `RidgeError::NonPositiveRegularization` |
| Singular / indefinite Gram matrix in Cholesky | Numerical refusal | `RidgeError::CholeskyIndefinite` |
| Empty calibration activation stream $\tau$ | Invariant violation | `SealError::EmptyCalibration` |
| Zero total counts in count table | Degenerate backoff | `CountLMError::EmptyContext` |
| Invalid trit code (>2 in 2-bit field) | Parse / unpack error | `TritError::InvalidCode` |

#### Evidence plan

1. **Positive tests:** focused tests for each mechanism (`cargo test --release --locked --test learn_<x>`).
2. **Negative controls:** per-task mutation controls executed in throwaway copies verifying test failure.
3. **Lean proofs:** kernel check `cd proof/corpus && lake build` for refinement theorems linking extracted carriers.
4. **Speed guard:** `python3 tools/gpu/speed_guard.py` verifying no inference throughput regression on golden model.

---

#### Mechanism specifications

##### (a) Seal quantizer (activation-calibrated, error $\sum \tau_j r_j^2$)

1. **Subsumed source:**
   - **Measured winner:**
     - Repository: `../spinor-ttl` (HEAD revision `a05bb1b`). File: `src/spinor_ttl/seal.py`, sha256 `bd012f144c6a584f1019e8d9e20ed756ddc6744063fa128130c2168cf998f996`. Functions: `aware_scale`, `function_distortion`, `hadamard`, `fit_blocks`, `quant_mag`, `_quant_blocks`, `fit_dials`, `quantize_tensor_song`, `seal_model`, `load_sealed`.
     - Repository: `../fold` (HEAD revision `d497161f`). File: `modelfold/src/quant.rs` (lines 1399–1825, `modelfold::quant::seal`), sha256 `4042468783b41ccf6ac6dbf0d68966dbfa02c91c54ab8bd30f1e7e75a612cf62`. Functions: `aware_scale`, `function_distortion`, `Seal::new`, `Seal::quantize`, `Seal::quantize_blocks`, `Seal::pack`, `Seal::unpack`.
   - **Predecessor:**
     - Repository: `../fold` (revision `e037ba11` / pre-consolidation `c47012be^`). File: `modelfold/src/seal.rs`, sha256 `35d21291cf17259b5032eab6cdd89457df2ed832d1a4308fe9093dc1da0c564a`.
2. **Baseline measurement:**
   - Exact benchmark commands: `python3 bench/vs_gguf.py` and `python3 bench/model_bits_token.py` in `../spinor-ttl`.
   - Measurement numbers recorded in `../spinor-ttl/PAPER.md` §3 (lines 114–128) and `../spinor-ttl/STATUS.md` (line 122):
     - Qwen2.5-0.5B Linear projections, 1024 held-out tokens with disjoint calibration band:
       - Seal reaches **4.2665 bits/token** at 4.45 bits/weight; `bitsandbytes` NF4 reaches 4.3032 bits/token at 4.50 bits/weight (`PAPER.md` lines 116–117).
       - Seal reaches **4.2057 bits/token** at 5.94 bits/weight; llama.cpp Q4_K_M reaches 4.2121 (no imatrix) and 4.2191 (imatrix) at 6.34 bits/weight (`PAPER.md` lines 122–124).
   - Grade: measured there, not re-run.
3. **Theorem to refine:**
   - `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:304`:
     ```lean
     theorem output_error_is_function_distortion {T n : ℕ} (X : Fin T → Fin n → ℝ)
         (r : Fin n → ℝ) (horth : ∀ j k, j ≠ k → ∑ t, X t j * X t k = 0) :
         ∑ t, (∑ j, X t j * r j) ^ 2
           = functionDistortion (fun j => ∑ t, X t j ^ 2) r
     ```
   - `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:333`:
     ```lean
     theorem aware_scale_is_the_chart {n : ℕ} (τ r : Fin n → ℝ) (hτ : ∀ j, 0 ≤ τ j) :
         functionDistortion τ r = ∑ j, (Real.sqrt (τ j) * r j) ^ 2
     ```
   - `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:344`:
     ```lean
     theorem weight_only_is_dual_blind {n : ℕ} (r : Fin n → ℝ) :
         (∀ τ τ' : Fin n → ℝ, functionDistortion τ r - functionDistortion τ' r
             = ∑ j, (τ j - τ' j) * r j ^ 2)
         ∧ (∀ j₀ : Fin n, r j₀ ≠ 0 → ∀ C : ℝ, ∃ τ : Fin n → ℝ,
             (∀ j, 0 ≤ τ j) ∧ C < functionDistortion τ r)
     ```
4. **Admissible domain:**
   - **Exact:** block length `blk = 32`, code bit width `bits = 4`, discrete quantized code values $q_j \in \{0, \dots, 2^B - 1\}$, integer bitstream packing / unpacking with format tag `HTSEAL1`. Aeneas-extractable integer domain.
   - **Float:** per-channel calibration energy $\tau_j = \sum_t X_{tj}^2$, scale factor $s_j = \sqrt{\tau_j}$, min/max range $d, m \in \mathbb{R}$, function distortion loss evaluation. Float operations tested against exact analytical references, not proved.
   - **Representation relation:** Rust `Seal` struct and packed tensor buffer mapped to Lean `functionDistortion τ r` over `Fin n → ℝ` in `MLPrimitivesBridge.lean`.
5. **Lineage search:**
   - Command: `rg -c "aware_scale" /home/socol/Workspace/{sanguine,sanguine-apeiron,fold,spinor-ttl,apeiron}`
   - Summary: 16 matching files found across all 5 repositories:
     - `spinor-ttl`: 6 files (`src/spinor_ttl/seal.py`, `PAPER.md`, `README.md`, `bench/seal_8b.py`, `bench/ternary_8b_vs_bnb.py`, `TERNARY_FINDINGS.md`).
     - `sanguine-apeiron`: 3 files (`theory/HEADLINES.tsv`, `proof/Apeiron/Audit.lean`, `proof/Apeiron/Numeric/Accumulator.lean`).
     - `sanguine`: 3 files (`proof/Substrate/Algebra/Neural/MLPrimitivesBridge.lean`, `release/paper1/Numerics.lean`, `release/paper1/INDEX.md`).
     - `fold`: 1 file (`modelfold/src/quant.rs`).
     - `apeiron`: 3 files (`docs/epics/neural-rebase.md`, `proof/corpus/AxiomAudit/Core.lean`, `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean`).
6. **Rust target:**
   - Proposed module: `src/learn/seal.rs` (with `src/learn/mod.rs` and `src/lib.rs`).
   - Test file: `tests/learn_seal.rs`.
   - Gate: `cargo test --release --locked --test learn_seal`.
   - Negative control (verbatim from `docs/plan.md`): drop the τ weighting in a copy: the calibrated-error ≤ uncalibrated-error test fails.
   - Metric: bits/token on Qwen2.5-0.5B $\le 4.2057$ at 5.94 bits/weight and $\le 4.2665$ at 4.45 bits/weight; calibration: 512 tokens, held-out: 1024 disjoint tokens.

---

##### (b) FoldAdapter (forward-only adapter, additive merge, exact unlearning)

1. **Subsumed source:**
   - **Measured winner:**
     - Repository: `../spinor-ttl` (HEAD revision `a05bb1b`). File: `src/spinor_ttl/finetune.py`, sha256 `02f2a669ab7b048561516b5829392d78bfc031fccc71839a09db4c90af73ce2b`. Functions: `FoldAdapter.__init__`, `FoldAdapter._fold`, `FoldAdapter.fold`, `FoldAdapter.unfold`, `FoldAdapter.discounts`, `FoldAdapter.refine`, `FoldAdapter._lane_tensors`, `FoldAdapter.refine_batch`, `FoldAdapter.bits_on`.
     - Repository: `../fold` (HEAD revision `d497161f`). File: `modelfold/src/train.rs` (lines 768–1050), sha256 `d685f8b1c1f47e4271c132ba8cb1ab08b93cd8b904dfbe12abb85459cffa994c`. Functions: `FoldAdapter::new`, `FoldAdapter::fold`, `FoldAdapter::unfold`, `FoldAdapter::discounts`, `FoldAdapter::refine`, `FoldAdapter::refine_with`, `FoldAdapter::refine_sparse`, `FoldAdapter::signature`.
   - **Predecessor:**
     - Repository: `../fold` (revision `f875705d`). File: `hashtrinity/src/finetune.rs`, sha256 `a532233529a9302a9e89b1580b23bec34c406e0720c657b78e104af1ef7e3424`; pre-consolidation `c47012be^` `modelfold/src/finetune.rs`, sha256 `d4b2f48c157fc1311f64f52b4f5eda0b8685196764855c6436f8f73ffbeb04cd`.
2. **Baseline measurement:**
   - Exact benchmark command: `python3 bench/finetune_race.py` in `../spinor-ttl`.
   - Measurement numbers recorded in `../spinor-ttl/PAPER.md` §4 (line 136) and `../spinor-ttl/STATUS.md` (lines 121–122):
     - Finetune fold⊕base reaches **0.724 b/B** in 2.9 s against PEFT LoRA r=16 **0.850 b/B** at 30 s (0.741 b/B at 300 s); fold alone reaches 0.957 b/B; frozen base reaches 0.910 b/B. Exact unlearning: `fold(A+B) − fold(B) = fold(A)`.
   - Grade: measured there, not re-run.
3. **Theorem to refine:**
   - `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:174`:
     ```lean
     theorem osFuse_is_add (a b : ℝ × ℝ) : osVal (osFuse a b) = osVal a + osVal b
     ```
   - `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:182`:
     ```lean
     theorem osFuse_grouping_free (a b c : ℝ × ℝ) :
         osVal (osFuse (osFuse a b) c) = osVal a (osFuse b c)
     ```
     *(Historical citation note: cited as `SpinorFold.lean:osFuse_*` in older papers due to historical namespace migration; in-repo declarations live in `MLPrimitivesBridge.lean`).*
   - `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:385`:
     ```lean
     theorem kn_floor_refinement {V : ℕ} (p : Fin V → ℝ) (c : Fin V → ℕ) (D : ℕ → ℝ)
         (hD0 : D 0 = 0) (hDb : ∀ n, 0 ≤ D n ∧ D n ≤ n) (hp0 : ∀ j, 0 ≤ p j)
         (hp1 : ∑ j, p j = 1) (hc : 0 < ∑ j, (c j : ℝ)) :
         (∀ j, 0 ≤ (∑ i, D (c i) / ∑ i, (c i : ℝ)) * p j + ((c j : ℝ) - D (c j)) / ∑ i, (c i : ℝ))
         ∧ (∑ j, ((∑ i, D (c i) / ∑ i, (c i : ℝ)) * p j + ((c j : ℝ) - D (c j)) / ∑ i, (c i : ℝ)) = 1)
     ```
   - `proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:101`:
     ```lean
     theorem forget_hebb (obs : K) (M : Memory K) : forget obs (hebb obs M) = M
     ```
4. **Admissible domain:**
   - **Exact:** context token IDs (`u32`), context order $k \le 4$, packed context keys (`u128`), integer counts $c_j \in \mathbb{N}$, integer count additions (`hebb`) and exact unlearning via subtraction (`forget`). Aeneas-extractable.
   - **Float:** absolute discounts $D_1, D_2, D_3 \in \mathbb{R}$, probability refinement blending mass $\gamma = \sum_i D(c_i)/\text{tot}$, and next-token distribution calculation $q_j = \gamma p_j + (c_j - D(c_j))/\text{tot}$. Tested in `tests/learn_foldadapt.rs`.
   - **Representation relation:** Rust count table `FreqMap` / `Table` mapped to Lean `Memory K` (`K → ℤ`) in `MirasCountLM.lean`, and refined distribution mapped to `kn_floor_refinement` in `MLPrimitivesBridge.lean`.
5. **Lineage search:**
   - Command: `rg -c "FoldAdapter" /home/socol/Workspace/{sanguine,sanguine-apeiron,fold,spinor-ttl,apeiron}`
   - Summary: 22 matching files found across repositories:
     - `spinor-ttl`: 6 files (`STATUS.md`, `PAPER.md`, `README.md`, `src/spinor_ttl/finetune.py`, `src/spinor_ttl/__init__.py`, `tests/test_finetune_refine.py`).
     - `fold`: 10 files (`modelfold/src/train.rs`, `modelfold/src/countlm.rs`, `modelfold/src/lib.rs`, `opsfold/src/flowops.rs`, `bedrock/examples/ttl_fact.rs`, `bedrock/examples/finetune_demo.rs`, `bedrock/examples/table_scaling.rs`, `modelfold/examples/cer_investigation.rs`, `modelfold/examples/gemma3_bench.rs`, `transcode/examples/codec_fidelity.rs`).
     - `sanguine`: 1 file (`theory/THE_GPU_TRINITY.md`).
     - `apeiron`: 5 files (`docs/plan.md`, `docs/epics/neural-rebase.md`, call-census tables).
6. **Rust target:**
   - Proposed module: `src/learn/foldadapt.rs`.
   - Test file: `tests/learn_foldadapt.rs`.
   - Gate: `cargo test --release --locked --test learn_foldadapt`.
   - Negative control (verbatim from `docs/plan.md`): weight each merge by its position in a copy: the grouping-free test fails.
   - Metric: b/B on finetune race corpus: $\le 0.724$ b/B against frozen base (0.910 b/B).

---

##### (c) Ridge fold (streaming Gram $X^T X$, one SPD solve, $U = X^T X + \lambda I$)

1. **Subsumed source:**
   - **Measured winner:**
     - Repository: `../spinor-ttl` (HEAD revision `a05bb1b`). Files: `src/spinor_ttl/foldtrain.py`, sha256 `a8b53c848a1c1610230ea514e16236683140625b0b1c083b739631ae8b8df47e`; and `src/spinor_ttl/ridgefold.py`, sha256 `c616adf61276251f37f5ce12192a818a78400b9fc60ae4ac73530280aad6c526`. Functions: `fold_ridge`, `FoldTrainer`, `fold_logistic`, `RidgeFold.fold`, `RidgeFold.unfold`, `RidgeFold.solve`, `RidgeFold.logit_delta`.
     - Repository: `../fold` (HEAD revision `d497161f`). File: `modelfold/src/foldtrain.rs`, sha256 `a5ff02895756a1373bcf7e5ade4725f1ebeece4b157293a8b40ed0c2b33c71a1`. Functions: `FoldTrainer::new`, `FoldTrainer::fit`, `FoldTrainer::fold`, `FoldTrainer::unfold`, `FoldTrainer::merge`, `FoldTrainer::solve`, `solve_spd`, `cholesky`, `chol_solve`, `fit_logistic`.
   - **Predecessor:**
     - Repository: `../fold` (commit `eaeca005`). File: `hashtrinity/src/ridgefold.rs`, sha256 `02a11e83ba5059ce466ded1dfcefd3a8963aeae7465b4d33f03e080594f3ec83`.
     - Repository: `../fold` (commit `db5397b2`). File: `hashtrinity/src/foldtrain.rs`, sha256 `6edb2705125f5b44ab338d59b56c1103b8e01e8a4804771cec949be36db555a8`.
2. **Baseline measurement:**
   - Exact benchmark command: `python3 bench/train_fold_vs_grad.py` in `../spinor-ttl`.
   - Measurement numbers recorded in `../spinor-ttl/STATUS.md` (lines 40–44):
     - RTX 5080, fitting the same ridge head to the same data, both landing on the same optimum ($\max |\Delta W| \le 10^{-2}$):
       - 200K rows, $d=512$: fold 24 ms (1 pass) vs Adam 418 ms (375 iters), **18× speedup**.
       - 500K rows, $d=1024$: fold 28 ms (1 pass) vs Adam 1952 ms (425 iters), **71× speedup**.
       - 1M rows, $d=256$: fold 59 ms (1 pass) vs Adam 928 ms (375 iters), **16× speedup**.
   - Grade: measured there, not re-run.
3. **Theorem to refine:**
   - `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean:223`:
     ```lean
     theorem ridgeHead_unique (X : Matrix (Fin n) (Fin d) ℝ) (Y : Matrix (Fin n) (Fin e) ℝ)
         {lam : ℝ} (hl : 0 < lam) (W' : Matrix (Fin d) (Fin e) ℝ)
         (hW' : drag X lam * W' = Xᵀ * Y) :
         W' = ridgeHead X Y lam
     ```
   - `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean:245`:
     ```lean
     theorem gram_stack (X₁ : Matrix (Fin n) (Fin d) ℝ) (X₂ : Matrix (Fin m) (Fin d) ℝ) :
         (stack X₁ X₂)ᵀ * stack X₁ X₂ = X₁ᵀ * X₁ + X₂ᵀ * X₂
     ```
   - `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean:251`:
     ```lean
     theorem boundary_stack (X₁ : Matrix (Fin n) (Fin d) ℝ) (X₂ : Matrix (Fin m) (Fin d) ℝ)
         (Y₁ : Matrix (Fin n) (Fin e) ℝ) (Y₂ : Matrix (Fin m) (Fin e) ℝ) :
         (stack X₁ X₂)ᵀ * stack Y₁ Y₂ = X₁ᵀ * Y₁ + X₂ᵀ * Y₂
     ```
4. **Admissible domain:**
   - **Exact:** integer/fixed-point Gram accumulation $(X_1 \oplus X_2)^T (X_1 \oplus X_2) = X_1^T X_1 + X_2^T X_2$, feature dimension $d$, sample counts $n$, shard accumulation monoid morphism. Aeneas-extractable.
   - **Float:** Cholesky decomposition $L L^T = U$, forward/backward substitution solving $U W = B$, regularization damping $\lambda > 0$. Tested in `tests/learn_ridge.rs`.
   - **Representation relation:** Rust Gram buffer `[f64; d*d]` and normal equations mapped to Lean `drag X lam` and `ridgeHead X Y lam` in `TrainingCollapse.lean`.
5. **Lineage search:**
   - Command: `rg -c "ridgeHead" /home/socol/Workspace/{sanguine,sanguine-apeiron,fold,spinor-ttl,apeiron}`
   - Summary: 56 matching files across the 5 repositories:
     - `spinor-ttl`: 4 files (`STATUS.md`, `src/spinor_ttl/foldtrain.py`, `bench/train_fold_vs_grad.py`, `bench/test_sparse_ridge_fold.py`).
     - `sanguine`: 24 files (`TrainingCollapse.lean`, `MLPrimitivesBridge.lean`, `LocalSupportRidge.lean`, `Train.lean`, `Model.lean`, etc.).
     - `fold`: 18 files (`modelfold/src/foldtrain.rs`, `modelfold/src/train.rs`, `modelfold/src/arch.rs`, `modelfold/tests/foldtrain.rs`, `bedrock/src/serve/ridge.rs`, etc.).
     - `apeiron`: 10 files (`docs/plan.md`, `docs/epics/neural-rebase.md`, `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean`, etc.).
6. **Rust target:**
   - Proposed module: `src/learn/ridge.rs`.
   - Test file: `tests/learn_ridge.rs`.
   - Gate: `cargo test --release --locked --test learn_ridge`.
   - Negative control (verbatim from `docs/plan.md`): drop λ in a copy: the unique-optimum test fails on a rank-deficient X.
   - Metric: Training wall-clock time to Adam optimum on $n=200\text{K}, d=512$: $\ge 16\times$ speedup over gradient descent baseline; residual MSE.

---

##### (d) KN count-LM (Kneser–Ney byte count model)

1. **Subsumed source:**
   - **Measured winner:**
     - Repository: `../spinor-ttl` (HEAD revision `a05bb1b`). File: `src/spinor_ttl/countlm.py`, sha256 `92b83507127b3c182dcc69c96569e917c05e2e31f966ea77cc0bec391551fe76`. Functions: `CountLM.__init__`, `CountLM.fit_stream`, `CountLM.bits_on`, `CountLM.predict`, `kn_discounts`.
     - Repository: `../fold` (HEAD revision `d497161f`). File: `modelfold/src/countlm.rs`, sha256 `cb7c9b3d2681829f51d30eeda81f9f63a25e0a29aec9c1f62d814d71eea34407`. Functions: `CountLM::new`, `CountLM::with_capacity`, `CountLM::fold`, `CountLM::merge`, `CountLM::unfold`, `CountLM::save`, `CountLM::load`, `CountLM::reader`, `kn_discounts_cg`, `Reader::dist`, `Reader::bits_per_byte`.
   - **Predecessor:**
     - Repository: `../fold` (commit `3c03ff38`). File: `hashtrinity/src/countlm.rs`, sha256 `c74f27696954fa464b50360470fdf5761d34900e7acd51e411e685491312859b`.
2. **Baseline measurement:**
   - Exact benchmark command: `python3 bench/beats_transformer.py` in `../spinor-ttl`.
   - Measurement numbers recorded in `../spinor-ttl/PAPER.md` §2 (lines 80–87) and `../spinor-ttl/STATUS.md` (line 120):
     - KN byte count-LM reaches **2.205 b/B** in 1.7 s against a 2-layer byte transformer reaching **2.392 b/B** in 20.9 s; the Lean `sang` binary reference agrees (**2.239 b/B** vs **2.469 b/B**).
   - Grade: measured there, not re-run.
3. **Theorem to refine:**
   - `proof/corpus/Substrate/Algebra/Neural/KneserNey.lean:98`:
     ```lean
     theorem kn_normalizes (c : Fin V → ℕ) (pLower : Fin V → ℝ) (D : ℝ)
         (hD0 : 0 ≤ D) (hD1 : D ≤ 1) (hp : ∑ w, pLower w = 1) (hc : 0 < total c) :
         ∑ w, pKN c pLower D w = 1
     ```
   - `proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:138`:
     ```lean
     theorem countlm_is_exact_miras (a b : K) (obs : K) (M : Memory K) (z : CrossUniverse.NCell) :
         (hebb a (hebb b M) = hebb b (hebb a M))
         ∧ (forget obs (hebb obs M) = M)
         ∧ (surprise (CrossUniverse.cellVolume z) = CrossUniverse.cellMass z)
     ```
   - `proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:89`:
     ```lean
     theorem hebb_comm (a b : K) (M : Memory K) : hebb a (hebb b M) = hebb b (hebb a M)
     ```
   - `proof/corpus/Substrate/Algebra/Neural/MirasCountLM.lean:101`:
     ```lean
     theorem forget_hebb (obs : K) (M : Memory K) : forget obs (hebb obs M) = M
     ```
4. **Admissible domain:**
   - **Exact:** Byte n-gram contexts, count table $c: \text{Fin } 256 \to \mathbb{N}$, order $k \le 4$, counts of counts $N_1, N_2, N_3, N_4 \in \mathbb{N}$, sovereign binary container `HTCLM01`. Aeneas-extractable.
   - **Float:** absolute discounts $D_k = k - (k+1) \frac{N_{k+1}}{N_k} \frac{N_1}{N_1 + 2 N_2}$, continuation probabilities $p_{KN} \in [0, 1]$, log-loss / b/B evaluation $- \log_2 p$. Tested in `tests/learn_count.rs`.
   - **Representation relation:** Rust `CountLM` state mapped to Lean `c : Fin V → ℕ` and `pKN c pLower D` in `KneserNey.lean`.
5. **Lineage search:**
   - Command: `rg -c "countlm_is_exact_miras|kn_normalizes" /home/socol/Workspace/{sanguine,sanguine-apeiron,fold,spinor-ttl,apeiron}`
   - Summary: 10 matching files found across repositories:
     - `sanguine`: 3 files (`proof/Substrate/Algebra/Neural/KneserNey.lean`, `MirasCountLM.lean`, `FrontierLadder.lean`).
     - `fold`: 4 files (`docs/SYNTHESIS.md`, `docs/STATE-ARCHIVE.md`, `STATE.md`, `modelfold/src/countlm.rs`).
     - `apeiron`: 3 files (`docs/plan.md`, `docs/epics/neural-rebase.md`, `proof/corpus/AxiomAudit/Core.lean`).
6. **Rust target:**
   - Proposed module: `src/learn/countlm.rs`.
   - Test file: `tests/learn_count.rs`.
   - Gate: `cargo test --release --locked --test learn_count`.
   - Negative control (verbatim from `docs/plan.md`): drop the KN discount in a copy: the normalisation test fails.
   - Metric: b/B on C4 / WikiText stream: $\le 2.205$ b/B; training 1M bytes, held-out 100K bytes.

---

##### (e) GPTQ / 4→3-trit planes

1. **Subsumed source:**
   - **Measured winner:**
     - Repository: `../fold` (HEAD revision `d497161f`). File: `modelfold/src/quant.rs` (lines 7–350, `modelfold::quant::gptq`), sha256 `4042468783b41ccf6ac6dbf0d68966dbfa02c91c54ab8bd30f1e7e75a612cf62`. Functions: `gptq_ternary`, `gptq_ternary_absmean`, `round_ternary`, `Quant::trit_planes`, `Quant::dequant_planes`, `output_error`, `gptq_affine`.
     - Repository: `../spinor-ttl` (HEAD revision `a05bb1b`). Files: `bench/ternary_8b_vs_bnb.py`, `bench/ternary_compute.py`, `bench/ternary_descent.py`.
   - **Predecessor:**
     - Repository: `../fold` (commit `d79460e8`). File: `hashtrinity/src/gptq.rs`, sha256 `8a02782deffd5a06ebfab16fc2b97b395c14efd52b02e976d60fe3f3f014c31b`; pre-consolidation `c47012be^` `modelfold/src/gptq.rs`, sha256 `17ad4d0b4cd1d418e60116964c765ffca56f823435bca9c3f78c0427999a1aad`.
2. **Baseline measurement:**
   - Exact benchmark commands: `cargo test --package hashtrinity --test gptq` (at `d79460e8`) and `cargo test --package modelfold --test gptq` (at `e037ba11`).
   - Measurement numbers recorded in `../fold` commit `d79460e8` (`hashtrinity/tests/gptq.rs`):
     - At $L=3$ (3 trits per weight, the 4-bit→ternary target), error-feedback reaches **$9.83 \times 10^{-4}$** output distortion vs naive rounding **$1.24 \times 10^{-3}$** (**$1.26\times$ distortion reduction**).
     - Balanced-trit plane sum reconstructs the affine dequant bit-for-bit (`balanced_ternary_lossless`).
   - Grade: measured there, not re-run.
3. **Theorem to refine:**
   - **NONE — stage-3 obligation.**
   - Missing statements:
     1. `gptq_error_feedback_distortion_le`: column-wise error feedback residual push under SPD Hessian $H = X^T X + \lambda I$ achieves strictly lower forward output distortion than naive rounding.
     2. `balanced_trit_plane_sum`: exact linear representation of affine code $c \in \{-13, \dots, +13\}$ via $\sum_{k=0}^{L-1} 3^k t_k$ for balanced trits $t_k \in \{-1, 0, +1\}$.
     *(Note: the distortion metric itself refines `proof/corpus/Substrate/Algebra/Neural/MLPrimitivesBridge.lean:304`, `output_error_is_function_distortion`).*
4. **Admissible domain:**
   - **Exact:** Balanced ternary values $\{-1, 0, +1\}$, 2-bit packing (4 trits/byte), radix-3 place-value reconstruction $\{1, 3, 9\}$, plane index. Aeneas-extractable.
   - **Float:** Hessian $H = X^T X + \lambda I$, Cholesky decomposition of $H^{-1}$, column error-feedback updates, output distortion metric. Tested in `tests/learn_gptq.rs`.
   - **Representation relation:** Rust `Q2Plane` (`Vec<u8>`) and place values mapped to Lean balanced ternary vectors `Fin L → Fin n → {-1, 0, 1}`.
5. **Lineage search:**
   - Command: `rg -c "gptq_ternary" /home/socol/Workspace/{sanguine,sanguine-apeiron,fold,spinor-ttl,apeiron}`
   - Summary: 6 matching files across repositories:
     - `spinor-ttl`: 1 file (`bench/ternary_descent.py`).
     - `fold`: 4 files (`modelfold/src/quant.rs`, `modelfold/src/arch.rs`, `modelfold/tests/gptq.rs`, `bedrock/src/serve/model/quant.rs`).
     - `apeiron`: 1 file (`docs/call-census/models.md`).
6. **Rust target:**
   - Proposed module: `src/learn/gptq.rs`.
   - Test file: `tests/learn_gptq.rs`.
   - Gate: `cargo test --release --locked --test learn_gptq`.
   - Negative control: zero the error-feedback residual push in a copy: the output-error < naive-round test fails.
   - Metric: Output distortion $\| X(W - \hat{W}) \|_F^2 \le 9.83 \times 10^{-4}$ on calibration activations $T=128, N=96$.

---

##### (f) FoldNet (deep, streaming, multimodal network trained by the closed-form fold)

1. **Subsumed source:**
   - **Measured winner:**
     - Repository: `../fold` (HEAD revision `d497161f`). File: `modelfold/src/arch.rs` (lines 14800+) and `modelfold/src/train.rs`, sha256 `d685f8b1c1f47e4271c132ba8cb1ab08b93cd8b904dfbe12abb85459cffa994c`.
   - **Predecessor:**
     - Repository: `../fold` (commit `83d9cc8a`). File: `hashtrinity/src/foldnet.rs`, sha256 `826bab489d685a47e54e5deba4e9af6ce59af277f721fa2d2bfdf34ee87c8984`; pre-consolidation `c47012be^` `modelfold/src/foldnet.rs`, sha256 `a1fd7171fa42dc3319ace820838ee48c46d29fc7ec140ee405920f0bc78d5089`. Functions: `RandomLayer::new`, `RandomLayer::forward`, `FoldNet::new`, `FoldNet::forward_features`, `FoldNet::fold`, `FoldNet::merge`, `FoldNet::solve`, `FoldNet::predict`, `FoldNet::quantize_head`.
2. **Baseline measurement:**
   - Exact benchmark commands: `cargo test --package hashtrinity --test foldnet` (at `83d9cc8a`) and `cargo run --package hashtrinity --example foldnet_demo`.
   - Measurement numbers recorded in `../fold` commit `83d9cc8a` (`hashtrinity/src/foldnet.rs`):
     - Rung 1 (Deep): XOR classification reaches **99.5% accuracy** (vs linear head 50.0%).
     - Rung 2 (Streaming): 40 streamed batches and 2 merged shards both give predictions identical to one whole fold (1000/1000).
     - Rung 3 (Multimodal): Two input widths (2-d, 5-d) into one head reach **98.4% accuracy**.
     - Arbitrary precision: Readout served at fp16/q4/tern3 holds XOR at **99.2%–99.5%**.
   - Grade: measured there, not re-run.
3. **Theorem to refine:**
   - Readout solve refines `proof/corpus/Substrate/Algebra/Neural/TrainingCollapse.lean:223` (`ridgeHead_unique`) and `:245` (`gram_stack`).
   - Fixed random nonlinear layer composition: **NONE — stage-3 obligation** (missing statement: `random_feature_ridge_composition`: composition of fixed feature map $\phi(x) = \tanh(W x + b)$ with unique ridge readout $\hat{W} = (\Phi^T \Phi + \lambda I)^{-1} \Phi^T Y$).
4. **Admissible domain:**
   - **Exact:** Network layer dimensions, random seed initialization, batch accumulation state counters, integer feature indexing. Aeneas-extractable.
   - **Float:** Pre-activation projection $W x + b$, nonlinear activation $\tanh$, floating-point Gram accumulation, Cholesky solve of the head. Tested in `tests/learn_foldnet.rs`.
   - **Representation relation:** Rust `FoldNet` readout mapped to `TrainingCollapse.ridgeHead` over feature map $\Phi$.
5. **Lineage search:**
   - Command: `rg -c "FoldNet" /home/socol/Workspace/{sanguine,sanguine-apeiron,fold,spinor-ttl,apeiron}`
   - Summary: 9 matching files across repositories:
     - `fold`: 9 files (`modelfold/src/arch.rs`, `foldnet_demo.rs`, `cmapss_rul_net.rs`, `cnn_fold.rs`, `glyph_ocr.rs`, `msx_foldnet.rs`, etc.).
6. **Rust target:**
   - Proposed module: `src/learn/foldnet.rs`.
   - Test file: `tests/learn_foldnet.rs`.
   - Gate: `cargo test --release --locked --test learn_foldnet`.
   - Negative control: zero the random layer bias or hidden dim in a copy: XOR classification accuracy drops to $\le 50\%$.
   - Metric: Classification accuracy on XOR benchmark $\ge 99.0\%$, multimodal accuracy $\ge 98.0\%$.

---

### Plan — Learning track (verified-AI sequence, stage 3)

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/neural-rebase.md`](neural-rebase.md)
- **Spec:** [`docs/epics/neural-rebase.md`](neural-rebase.md)
- **Status:** draft

```text
SCOPE: docs/epics/neural-rebase.md only; wave map and task definitions for learning track
REGISTER: learning track implementation and verification planning
CORPUS: proof/corpus/Substrate/Algebra/Neural/{MirasCountLM, KneserNey, CountLMFloor,
        AttentionSpinor, MLPrimitivesBridge, TrainingCollapse}.lean;
        proof/corpus/Substrate/Algebra/NullCell/SpinorFold.lean
REFERENCE: docs/plan.md Learning track; ../spinor-ttl; ../fold/modelfold
GATE: python3 tools/check-claims.py
GRADE: anchor/planning
```

#### 1. Goal and completion condition

Reimplement and verify the six learning track mechanisms in this crate under `src/learn/`, ordered hardest-and-highest-contribution first: seal quantizer, FoldAdapter, ridge fold, KN count-LM, GPTQ / 4→3-trit planes, and FoldNet. Each mechanism carries its test suite `tests/learn_<x>.rs`, passes its focused gate `cargo test --release --locked --test learn_<x>`, runs its negative mutation control, and links to its vendored Lean declaration under `proof/corpus/Substrate/Algebra/Neural/`.

#### 2. Non-goals and trust boundary

- Non-goals: Full-scale distributed cluster pre-training; novel architecture search; modifying the pinned canonical corpus at `../sanguine-apeiron/proof`; regression of inference decode speeds (`tg128` and `pp512` on Qwen2.5-0.5B).
- Trust boundary: Covers the in-tree Rust implementation, unit/integration test suites, Aeneas extraction models, and Lean kernel-checked refinement links. Does not cover host OS scheduling, GPU vendor drivers, or third-party python evaluation harnesses.

#### 3. Lane map and serialization rules

1. **Serialization on shared modules:** Shared root files `src/learn/mod.rs` and `src/lib.rs` are owned by the integrator lane. Worker tasks on disjoint mechanisms touch disjoint implementation files.
2. **Worktree isolation:** Each task executes in an isolated `worktree-agent-<id>` worktree.
3. **Execution bounds:** All Rust builds and tests run bounded: `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0` with `timeout 600`.

#### 4. Wave map

| Wave | Task ID | Owner | Owned files (≤3) | One focused gate | Negative control |
|---|---|---|---|---|---|
| W0 | `APEIRON-LEARN-W0-T1` | integrator | `src/learn/gguf.rs`, `src/learn/mod.rs`, `tests/learn_gguf.rs` | `cargo test --release --locked --test learn_gguf` | Flip magic byte in copy: parser fails |
| W0 | `APEIRON-LEARN-W0-T2` | integrator | `tools/gpu/speed_guard.py` | `python3 tools/gpu/speed_guard.py` | Add sleep in decode loop in copy: gate fails by name |
| W1 | `APEIRON-LEARN-W1-T1` | worker | `src/learn/seal.rs`, `tests/learn_seal.rs` | `cargo test --release --locked --test learn_seal` | Drop τ weighting in copy: calibrated-error ≤ uncalibrated-error fails |
| W2 | `APEIRON-LEARN-W2-T1` | worker | `src/learn/foldadapt.rs`, `tests/learn_foldadapt.rs` | `cargo test --release --locked --test learn_foldadapt` | Weight each merge by position in copy: grouping-free test fails |
| W3 | `APEIRON-LEARN-W3-T1` | worker | `src/learn/ridge.rs`, `tests/learn_ridge.rs` | `cargo test --release --locked --test learn_ridge` | Drop λ in copy: unique-optimum fails on rank-deficient X |
| W4 | `APEIRON-LEARN-W4-T1` | worker | `src/learn/countlm.rs`, `tests/learn_count.rs` | `cargo test --release --locked --test learn_count` | Drop the backoff term (keep the discount) in copy: normalisation test fails (mass < 1) |
| W5 | `APEIRON-LEARN-W5-T1` | worker | `src/learn/gptq.rs`, `tests/learn_gptq.rs` | `cargo test --release --locked --test learn_gptq` | Zero error-feedback push in copy: output-error < naive-round fails |
| W2 | `APEIRON-LEARN-W2-T2` | worker | `proof/extract/learn_merge.rs`, `proof/corpus/Learn/FoldMerge.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build Learn.FoldMerge AxiomAudit` | Make the extracted merge non-commutative in copy: refinement to `osFuse_is_add` fails |
| W4 | `APEIRON-LEARN-W4-T2` | worker | `proof/extract/learn_count.rs`, `proof/corpus/Learn/CountLM.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build Learn.CountLM AxiomAudit` | Off-by-one in the extracted count update in copy: refinement to `countlm_is_exact_miras` fails |
| W5 | `APEIRON-LEARN-W5-T2` | worker | `proof/extract/learn_trit.rs`, `proof/corpus/Learn/TritPlanes.lean`, `proof/corpus/AxiomAudit.lean` | `cd proof/corpus && lake build Learn.TritPlanes AxiomAudit` | Swap two trit planes in the extracted recombination in copy: exact-reconstruction theorem fails |
| W6 | `APEIRON-LEARN-W6-T1` | worker | `src/learn/foldnet.rs`, `tests/learn_foldnet.rs` | `cargo test --release --locked --test learn_foldnet` | Zero random bias in copy: XOR accuracy drops to 50% |

#### 5. Task blocks

##### APEIRON-LEARN-W0-T1 · GGUF write path and tensor container
- **Scope:** Implement GGUF container export in `src/learn/gguf.rs` capable of writing quantized tensor structures loaded by the inference engine.
- **Owned files:** `src/learn/gguf.rs`, `src/learn/mod.rs`, `tests/learn_gguf.rs`
- **Depends on:** none
- **Grade:** `tested`
- **Gate:** `cargo test --release --locked --test learn_gguf`
- **Negative control:** Invert the GGUF header magic in a disposable copy; parser verification must fail. Restore and verify exit 0.

##### APEIRON-LEARN-W0-T2 · Inference speed guard verification
- **Scope:** Establish `tools/gpu/speed_guard.py` measuring `tg128` and `pp512` token rates on Qwen2.5-0.5B Q4_K_M, verifying SEQHASH `9092dbbaff119a2e`.
- **Owned files:** `tools/gpu/speed_guard.py`
- **Depends on:** none
- **Grade:** `tested`
- **Gate:** `python3 tools/gpu/speed_guard.py`
- **Negative control:** Insert a 10 ms delay in the decode step in a disposable copy; speed guard must fail on threshold. Restore and verify exit 0.

##### APEIRON-LEARN-W1-T1 · Port and verify seal activation-calibrated quantizer (`LEARN-SEAL`)
- **Scope:** Port activation-calibrated quantizer minimizing $\sum \tau_j r_j^2$ from `../spinor-ttl/src/spinor_ttl/seal.py` and `../fold/modelfold/src/quant.rs`. Refines `MLPrimitivesBridge.output_error_is_function_distortion`.
- **Owned files:** `src/learn/seal.rs`, `tests/learn_seal.rs`
- **Depends on:** `APEIRON-LEARN-W0-T1`
- **Grade:** `tested`
- **Gate:** `cargo test --release --locked --test learn_seal`
- **Negative control:** Drop the τ weighting in a disposable copy; the calibrated-error ≤ uncalibrated-error test must fail. Restore and verify exit 0.

##### APEIRON-LEARN-W2-T1 · Port and verify FoldAdapter forward-only finetuning (`LEARN-FOLDADAPT`)
- **Scope:** Port forward-only FoldAdapter with exact additive unlearning and Kneser–Ney distribution blending from `../spinor-ttl/src/spinor_ttl/finetune.py`. Refines `MLPrimitivesBridge.osFuse_is_add`, `osFuse_grouping_free`, and `MirasCountLM.forget_hebb`.
- **Owned files:** `src/learn/foldadapt.rs`, `tests/learn_foldadapt.rs`
- **Depends on:** none (no GGUF output)
- **Grade:** `tested`
- **Gate:** `cargo test --release --locked --test learn_foldadapt`
- **Negative control:** Weight each merge by its position in a disposable copy; the grouping-free test must fail. Restore and verify exit 0.

##### APEIRON-LEARN-W3-T1 · Port and verify ridge fold streaming Gram and SPD solve (`LEARN-RIDGE`)
- **Scope:** Port streaming Gram accumulation $X^T X$ and direct Cholesky SPD solve from `../fold/modelfold/src/foldtrain.rs`. Refines `TrainingCollapse.ridgeHead_unique` and `gram_stack`.
- **Owned files:** `src/learn/ridge.rs`, `tests/learn_ridge.rs`
- **Depends on:** none (no GGUF output)
- **Grade:** `tested`
- **Gate:** `cargo test --release --locked --test learn_ridge`
- **Negative control:** Drop λ regularization in a disposable copy; the unique-optimum test must fail on a rank-deficient matrix $X$. Restore and verify exit 0.

##### APEIRON-LEARN-W4-T1 · Port and verify KN count-LM byte model (`LEARN-COUNT`)
- **Scope:** Port KN-smoothed byte count language model over the coordinate-keyed counting face from `../fold/modelfold/src/countlm.rs`. Refines `KneserNey.kn_normalizes` and `MirasCountLM.countlm_is_exact_miras`.
- **Owned files:** `src/learn/countlm.rs`, `tests/learn_count.rs`
- **Depends on:** none (no GGUF output)
- **Grade:** `tested`
- **Gate:** `cargo test --release --locked --test learn_count`
- **Negative control:** Drop the backoff term while keeping the discount in a disposable copy; the discounted mass is lost and the normalisation test must fail. (Dropping the discount alone leaves the maximum-likelihood estimate, which still normalises.) Restore and verify exit 0.

##### APEIRON-LEARN-W5-T1 · Port and verify GPTQ error feedback and balanced-trit planes (`LEARN-GPTQ`)
- **Scope:** Port 4-bit to 3-trit error-feedback post-hoc quantizer and balanced-trit plane decomposition from `../fold/modelfold/src/quant.rs`. Refines `MLPrimitivesBridge.output_error_is_function_distortion`.
- **Owned files:** `src/learn/gptq.rs`, `tests/learn_gptq.rs`
- **Depends on:** `APEIRON-LEARN-W3-T1`
- **Grade:** `tested`
- **Gate:** `cargo test --release --locked --test learn_gptq`
- **Negative control:** Zero the error-feedback residual push in a disposable copy; the output-error < naive-round test must fail. Restore and verify exit 0.

##### APEIRON-LEARN-W6-T1 · Port and verify FoldNet deep streaming multimodal learner (`LEARN-FOLDNET`)
- **Scope:** Port FoldNet deep fixed random feature expansion and streaming multimodal learner from `../fold/hashtrinity/src/foldnet.rs` and `modelfold/src/arch.rs`. Refines `TrainingCollapse.ridgeHead_unique`.
- **Owned files:** `src/learn/foldnet.rs`, `tests/learn_foldnet.rs`
- **Depends on:** `APEIRON-LEARN-W3-T1`
- **Grade:** `tested`
- **Gate:** `cargo test --release --locked --test learn_foldnet`
- **Negative control:** Zero the random layer bias in a disposable copy; XOR classification accuracy must collapse to $\le 50\%$. Restore and verify exit 0.

#### 6. Exit table

| Stage / Scope | Command | Target / Focus | Expected exit |
|---|---|---|---|
| W0 GGUF container | `cargo test --release --locked --test learn_gguf` | Serialization format | 0 |
| W0 Speed guard | `python3 tools/gpu/speed_guard.py` | Non-regression check | 0 |
| W1 Seal quantizer | `cargo test --release --locked --test learn_seal` | Activation-calibrated quantizer | 0 |
| W2 FoldAdapter | `cargo test --release --locked --test learn_foldadapt` | Forward-only finetuning | 0 |
| W3 Ridge fold | `cargo test --release --locked --test learn_ridge` | Streaming Gram and SPD solve | 0 |
| W4 KN count-LM | `cargo test --release --locked --test learn_count` | Byte count model | 0 |
| W5 GPTQ 4→3-trit | `cargo test --release --locked --test learn_gptq` | Error-feedback trit planes | 0 |
| W6 FoldNet | `cargo test --release --locked --test learn_foldnet` | Deep streaming learner | 0 |
| W2/W4/W5 refinements | `cd proof/corpus && lake build Learn.FoldMerge Learn.CountLM Learn.TritPlanes AxiomAudit` | Exact kernels extracted from the real Rust and refined, closed cone | 0 |
| **Stage 3 Barrier** | `cargo test --release --locked --test 'learn_*'` | Full learning test suite | **0** |

#### 7. Lead decisions

1. **GGUF:** the seal quantizer and GPTQ write standard GGUF directly, readable by the inference runner and by llama.cpp. No private container. Reversal: replace `APEIRON-LEARN-W0-T1`.
2. **Proof scope:** the exact integer kernels are extracted from the real Rust and refined. These are the FoldAdapter merge (`osFuse_is_add`), the integer Gram accumulator over quantized inputs (`gram_stack`), the count update (`countlm_is_exact_miras`) and the trit-plane decomposition (exact reconstruction): `APEIRON-LEARN-W{2,3,4,5}-T2`, grade `proved`. `W3-T2` takes the same shape as the other three: `proof/extract/learn_gram.rs`, `proof/corpus/Learn/Gram.lean`, gate `lake build Learn.Gram AxiomAudit`, control a dropped cross term in the extracted accumulator. The float paths (seal grid search, ridge solve, GPTQ column feedback, FoldNet) are `tested` against analytical references. The Lean theorems over ℝ that they instantiate are cited as `model` for the Rust, never as `proved`. A Lean statement of GPTQ column feedback stays an open obligation, not a task.
