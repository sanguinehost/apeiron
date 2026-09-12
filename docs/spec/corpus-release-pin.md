# Specification - Immutable Corpus Dependency

- **Owner:** OpenCode
- **Intent:** `docs/intent/corpus-release-pin.md`
- **Source revision:** `7cec238d7544a9914e8f71703a2c0c08bf9e0883`; working tree clean before this arc
- **Status:** approved

## Scope and admissible domain

| Item | Value |
|---|---|
| Operation | Lake dependency resolution for package `apeiron` |
| Domain | A checkout with Git access to the configured corpus repository and Lean 4.31.0 |
| Totality | External command; nonzero exit on unavailable or invalid dependency |
| Preconditions | The exact commit exists and contains the Lake package in `proof` |
| Postcondition | The manifest records the Git URL, full input revision, resolved revision and `proof` subdirectory |

No fallback to the sibling path is permitted. Missing credentials, repository access or commit
availability is a visible build failure.

## Artifact relation

| Layer | Artifact | Path |
|---|---|---|
| Canonical source | Corpus commit | `d9aefc93522e2109bc96158526de6dbd52477276` |
| Dependency declaration | Git revision and `proof` subdirectory | `proof/corpus/lakefile.lean` |
| Resolved dependency | Git source and resolved commit | `proof/corpus/lake-manifest.json` |
| Mathematical root | `Apeiron.Trichotomy.AnyRing` | Corpus `proof/Apeiron/Trichotomy/AnyRing.lean` |
| Refinement consumer | `WordRefinement` | `proof/corpus/WordRefinement.lean` |

The package manager checks out the canonical source. The bridge continues to import declarations by
their canonical names; no theorem is copied into Apeiron.

## Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| Exact source identity | `inputRev` and `rev` equal the full corpus commit. | tested | Manifest assertion. |
| Package identity | Dependency type is `git` and `subDir` is `proof`. | tested | Manifest assertion. |
| Compatibility | The complete bridge builds with the pinned package. | proved | `lake build` in `proof/corpus`. |
| Existing proof freshness | The generated Rust extraction is unchanged and current. | tested | `proof/regen.sh --check`. |

## Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Missing repository access | Nonzero `lake update` | Git transport failure |
| Missing or invalid commit | Nonzero `lake update` | Revision resolution failure |
| Missing `proof` package | Nonzero Lake load | Package subdirectory failure |
| Incompatible corpus | Nonzero `lake build` | Lean elaboration failure |
| Mutable or local corpus declaration | Nonzero manifest assertion | Source identity failure |

## Evidence plan

1. Run the manifest assertion before implementation and record its failure on the path dependency.
2. Change only the Lake dependency declaration and regenerate the manifest with `lake update apeiron`.
3. Assert the manifest records the full revision, Git type and `proof` subdirectory.
4. Run `lake build` in `proof/corpus` and `proof/regen.sh --check` from the repository root.
5. In a throwaway copy, replace the commit with a nonexistent revision and confirm
   `lake update apeiron` exits nonzero; retain the production tree unchanged.

## Trust boundary

This specification establishes dependency identity and successful Lean checking. It does not prove
the package manager, Git server, transport, Lean implementation, compiler backend or hardware.

## Open questions

- None.
