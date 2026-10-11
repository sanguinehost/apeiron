# Immutable Corpus Dependency

Merged epic record: `intent/corpus-release-pin.md`, `spec/corpus-release-pin.md`, `plan/corpus-release-pin.md` (one file per epic; see `docs/README.md`).

## Intent - Immutable Corpus Dependency

- **Owner:** OpenCode
- **Source revision:** `7cec238d7544a9914e8f71703a2c0c08bf9e0883`; working tree clean
- **Status:** approved
- **Spec:** `docs/epics/corpus-release-pin.md`
- **Plan:** `docs/epics/corpus-release-pin.md`

### Problem

The refinement bridge loads the canonical mathematical corpus through
`../../../sanguine/proof`. A sibling path can resolve to any working-tree state, so a clean
Apeiron checkout does not identify the corpus source checked by `lake build`.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Retain the sibling path and record a commit in prose | Lake still builds the unchecked sibling state. |
| Copy selected theorem statements into the bridge | This creates a second mathematical authority and breaks literal trichotomy rooting. |
| Vendor only the current import closure | The selected closure can drift from the canonical package and its fail-closed audits. |
| Pin the repository branch name | A branch is mutable and does not identify release input. |

### Completion condition

Lake resolves package `apeiron` from the published repository at the full corpus commit, loads its
`proof` package, and builds the bridge.

```sh
sh -c 'cd proof/corpus && lake update apeiron && lake build'
```

### Invariants

1. The bridge imports canonical declarations from `Apeiron.Trichotomy.AnyRing`; it does not restate
   their formulas.
2. The corpus, bridge, Aeneas backend and Mathlib use Lean 4.31.0-compatible revisions.
3. The local extraction package remains a path dependency because it is part of this repository.
4. No corpus source or theorem statement changes in this epic.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| Lake selects corpus commit `d9aefc93522e2109bc96158526de6dbd52477276`. | tested | Generated `proof/corpus/lake-manifest.json` assertion. |
| The pinned corpus and bridge elaborate together. | proved | `lake build` in `proof/corpus`. |
| Existing extracted Rust remains fresh. | tested | `proof/regen.sh --check`. |

### Non-goals

- Moving the corpus history into this repository.
- Changing a theorem, extracted Rust kernel or application integration.
- Vendoring Mathlib, Aeneas or Charon.
- Establishing availability of the private Git service without credentials.

### Trust boundary

The commit pin identifies source content. It does not verify Git transport, hosting availability,
rustc/LLVM, Aeneas translation, Lean binaries or the optimized storage layer.

### Open questions

- None for this arc. Corpus history convergence remains a separate repository-level migration.

## Specification - Immutable Corpus Dependency

- **Owner:** OpenCode
- **Intent:** `docs/epics/corpus-release-pin.md`
- **Source revision:** `7cec238d7544a9914e8f71703a2c0c08bf9e0883`; working tree clean before this arc
- **Status:** approved

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | Lake dependency resolution for package `apeiron` |
| Domain | A checkout with Git access to the configured corpus repository and Lean 4.31.0 |
| Totality | External command; nonzero exit on unavailable or invalid dependency |
| Preconditions | The exact commit exists and contains the Lake package in `proof` |
| Postcondition | The manifest records the Git URL, full input revision, resolved revision and `proof` subdirectory |

No fallback to the sibling path is permitted. Missing credentials, repository access or commit
availability is a visible build failure.

### Artifact relation

| Layer | Artifact | Path |
|---|---|---|
| Canonical source | Corpus commit | `d9aefc93522e2109bc96158526de6dbd52477276` |
| Dependency declaration | Git revision and `proof` subdirectory | `proof/corpus/lakefile.lean` |
| Resolved dependency | Git source and resolved commit | `proof/corpus/lake-manifest.json` |
| Mathematical root | `Apeiron.Trichotomy.AnyRing` | Corpus `proof/Apeiron/Trichotomy/AnyRing.lean` |
| Refinement consumer | `WordRefinement` | `proof/corpus/WordRefinement.lean` |

The package manager checks out the canonical source. The bridge continues to import declarations by
their canonical names; no theorem is copied into Apeiron.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| Exact source identity | `inputRev` and `rev` equal the full corpus commit. | tested | Manifest assertion. |
| Package identity | Dependency type is `git` and `subDir` is `proof`. | tested | Manifest assertion. |
| Compatibility | The complete bridge builds with the pinned package. | proved | `lake build` in `proof/corpus`. |
| Existing proof freshness | The generated Rust extraction is unchanged and current. | tested | `proof/regen.sh --check`. |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Missing repository access | Nonzero `lake update` | Git transport failure |
| Missing or invalid commit | Nonzero `lake update` | Revision resolution failure |
| Missing `proof` package | Nonzero Lake load | Package subdirectory failure |
| Incompatible corpus | Nonzero `lake build` | Lean elaboration failure |
| Mutable or local corpus declaration | Nonzero manifest assertion | Source identity failure |

### Evidence plan

1. Run the manifest assertion before implementation and record its failure on the path dependency.
2. Change only the Lake dependency declaration and regenerate the manifest with `lake update apeiron`.
3. Assert the manifest records the full revision, Git type and `proof` subdirectory.
4. Run `lake build` in `proof/corpus` and `proof/regen.sh --check` from the repository root.
5. In a throwaway copy, replace the commit with a nonexistent revision and confirm
   `lake update apeiron` exits nonzero; retain the production tree unchanged.

### Trust boundary

This specification establishes dependency identity and successful Lean checking. It does not prove
the package manager, Git server, transport, Lean implementation, compiler backend or hardware.

### Open questions

- None.

## Plan - Immutable Corpus Dependency

- **Owner:** OpenCode
- **Intent:** `docs/epics/corpus-release-pin.md`
- **Spec:** `docs/epics/corpus-release-pin.md`
- **Source revision:** `7cec238d7544a9914e8f71703a2c0c08bf9e0883`; working tree clean before this arc
- **Status:** approved

### Goal and completion condition

The bridge resolves and builds the exact canonical corpus revision without reading the sibling
corpus worktree.

```sh
sh -c 'cd proof/corpus && lake update apeiron && lake build'
```

### Dependency position

```text
immutable corpus source
  -> storage reference and refinement
  -> additional extracted kernels
  -> Song and Penglai consumers
```

The corpus theorem backlog is independent of this package-resolution task but every Rust refinement
release depends on an immutable mathematical source.

### Non-goals

- Corpus theorem changes.
- Storage refinement.
- Song or Penglai changes.

### Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | Record the approved lifecycle | Three lifecycle files | OpenCode | `git diff --check` | Not applicable: planning artifacts change no executable claim. |
| W2 | Pin and resolve the corpus | `proof/corpus/lakefile.lean`, `proof/corpus/lake-manifest.json` | OpenCode | `lake build` in `proof/corpus` | Nonexistent commit makes `lake update apeiron` fail in a throwaway copy. |
| W3 | Record evidence and boundaries | `proof/README.md`, `docs/plan.md`, `docs/direction.md` | OpenCode | `git diff --check` | Removing the manifest source assertion from the evidence record does not close W2. |

### Task blocks

#### W1-T1 - Lifecycle records

- **Scope:** Commit intent, specification and execution barriers before changing dependency code.
- **Files:** `docs/epics/corpus-release-pin.md`, `docs/epics/corpus-release-pin.md`,
  `docs/epics/corpus-release-pin.md`
- **Depends on:** None
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. These files make no execution or proof claim.
- **Evidence to record:** Starting revision and the failing manifest assertion.

#### W2-T1 - Exact corpus source

- **Scope:** Replace the sibling path with a full Git commit and build the resolved package.
- **Files:** `proof/corpus/lakefile.lean`, `proof/corpus/lake-manifest.json`
- **Depends on:** W1-T1
- **Grade:** proved
- **Gate:** `lake build` in `proof/corpus`
- **Negative control:** Replace the revision with a nonexistent commit in a throwaway copy;
  `lake update apeiron` must exit nonzero. Restore is unnecessary because the production tree is not
  modified by the control.
- **Evidence to record:** Git URL, input revision, resolved revision, subdirectory and build job count.

#### W3-T1 - Evidence record

- **Scope:** Replace the open release-pin statement with commands, revisions, result and boundary.
- **Files:** `proof/README.md`, `docs/plan.md`, `docs/direction.md`
- **Depends on:** W2-T1
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. W2 owns the executable dependency control.
- **Evidence to record:** Manifest assertion, mutation failure, proof build and extraction freshness.

### Barrier checklist

- [x] Every task gate exits 0 and the dependency mutation exits 128.
- [x] No task changes canonical corpus source or theorem statements.
- [x] `cargo test`, `cargo clippy` and `cargo fmt` pass.
- [x] `proof/regen.sh --check` reports fresh and `proof/corpus` builds.
- [x] Source and tool revisions are recorded.
- [x] Claims retain their existing evidence grades and trust boundaries.

### Acceptance result

| Command | Exit | Result |
|---|---|---|
| Manifest source assertion | 0 | Git source, full input/resolved revision and `proof` subdirectory match. |
| `lake update apeiron` | 0 | Corpus commit resolved from the configured Git service. |
| `lake build` in `proof/corpus` | 0 | 3276 jobs; fail-closed bridge audit passes. |
| Nonexistent-revision control | 128 | Git cannot read the zero revision; no local fallback occurs. |
| `proof/regen.sh --check` | 0 | Extraction fresh; 1708 jobs. |
| `cargo test --workspace --all-features --locked` | 0 | 164 unit/integration tests and 9 documentation tests pass. |
| `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings` | 0 | No findings. |
| `cargo fmt --all -- --check` | 0 | No formatting drift. |
| `cargo kani` | 0 | 60 of 60 harnesses verified. |
| `git diff --check` | 0 | No whitespace errors. |

### Stop conditions

- Stop this arc if the published commit cannot be fetched or does not expose the `proof` package.
- After three distinct package-resolution approaches fail, report the smallest unresolved source
  identity or package-layout requirement rather than restoring a mutable fallback.
