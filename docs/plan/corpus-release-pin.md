# Plan - Immutable Corpus Dependency

- **Owner:** OpenCode
- **Intent:** `docs/intent/corpus-release-pin.md`
- **Spec:** `docs/spec/corpus-release-pin.md`
- **Source revision:** `7cec238d7544a9914e8f71703a2c0c08bf9e0883`; working tree clean before this arc
- **Status:** approved

## Goal and completion condition

The bridge resolves and builds the exact canonical corpus revision without reading the sibling
corpus worktree.

```sh
sh -c 'cd proof/corpus && lake update apeiron && lake build'
```

## Dependency position

```text
immutable corpus source
  -> storage reference and refinement
  -> additional extracted kernels
  -> Song and Penglai consumers
```

The corpus theorem backlog is independent of this package-resolution task but every Rust refinement
release depends on an immutable mathematical source.

## Non-goals

- Corpus theorem changes.
- Storage refinement.
- Song or Penglai changes.

## Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | Record the approved lifecycle | Three lifecycle files | OpenCode | `git diff --check` | Not applicable: planning artifacts change no executable claim. |
| W2 | Pin and resolve the corpus | `proof/corpus/lakefile.lean`, `proof/corpus/lake-manifest.json` | OpenCode | `lake build` in `proof/corpus` | Nonexistent commit makes `lake update apeiron` fail in a throwaway copy. |
| W3 | Record evidence and boundaries | `proof/README.md`, `docs/flash-handoff.md`, `docs/direction.md` | OpenCode | `git diff --check` | Removing the manifest source assertion from the evidence record does not close W2. |

## Task blocks

### W1-T1 - Lifecycle records

- **Scope:** Commit intent, specification and execution barriers before changing dependency code.
- **Files:** `docs/intent/corpus-release-pin.md`, `docs/spec/corpus-release-pin.md`,
  `docs/plan/corpus-release-pin.md`
- **Depends on:** None
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. These files make no execution or proof claim.
- **Evidence to record:** Starting revision and the failing manifest assertion.

### W2-T1 - Exact corpus source

- **Scope:** Replace the sibling path with a full Git commit and build the resolved package.
- **Files:** `proof/corpus/lakefile.lean`, `proof/corpus/lake-manifest.json`
- **Depends on:** W1-T1
- **Grade:** proved
- **Gate:** `lake build` in `proof/corpus`
- **Negative control:** Replace the revision with a nonexistent commit in a throwaway copy;
  `lake update apeiron` must exit nonzero. Restore is unnecessary because the production tree is not
  modified by the control.
- **Evidence to record:** Git URL, input revision, resolved revision, subdirectory and build job count.

### W3-T1 - Evidence record

- **Scope:** Replace the open release-pin statement with commands, revisions, result and boundary.
- **Files:** `proof/README.md`, `docs/flash-handoff.md`, `docs/direction.md`
- **Depends on:** W2-T1
- **Grade:** tested
- **Gate:** `git diff --check`
- **Negative control:** Not applicable. W2 owns the executable dependency control.
- **Evidence to record:** Manifest assertion, mutation failure, proof build and extraction freshness.

## Barrier checklist

- [x] Every task gate exits 0 and the dependency mutation exits 128.
- [x] No task changes canonical corpus source or theorem statements.
- [x] `cargo test`, `cargo clippy` and `cargo fmt` pass.
- [x] `proof/regen.sh --check` reports fresh and `proof/corpus` builds.
- [x] Source and tool revisions are recorded.
- [x] Claims retain their existing evidence grades and trust boundaries.

## Acceptance result

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

## Stop conditions

- Stop this arc if the published commit cannot be fetched or does not expose the `proof` package.
- After three distinct package-resolution approaches fail, report the smallest unresolved source
  identity or package-layout requirement rather than restoring a mutable fallback.
