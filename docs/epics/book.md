# Public book

Merged epic record: intent, specification and plan in one file (see `docs/README.md`).

## Intent — Public book

- **Owner:** lead
- **Source revision:** `ce06316`; working tree dirty in files owned by other streams only
- **Status:** complete
- **Spec:** `docs/epics/book.md`
- **Plan:** `docs/epics/book.md`

### Problem

The public release has no document a beginner can follow to a running `Cell`, `Table` and model, and
none that an expert can check against the proofs. `docs/direction.md`, `docs/kernel-boundary.md`,
`proof/README.md` and `docs/hash-comparison.md` are programme records: they name internal lanes, plan
codes and historical layouts, and run to 3000 lines between them.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Ship the four programme documents | They carry process vocabulary and historical layouts; a reader cannot tell the executing table from a retired one. |
| One long README | Mixes install, theory and verification; no executable examples. |
| rustdoc only | Covers symbols, not the theory or the evidence grades. |
| Prose without compiled examples | An example that does not compile is a defect the reader finds first. |

### Completion condition

`book/` builds, every Rust block in it compiles against the crate at the checked-out revision, and
every path or declaration it cites resolves.

```sh
mdbook build book && mdbook test book -L target/debug/deps && python3 tools/check-claims.py && python3 tools/book-check.py
```

### Invariants

1. Every Rust block is compiled by `mdbook test`; a block that is not meant to compile is marked
   `ignore` or `text` with a reason in the sentence before it.
2. Every displayed equation carries a glossary: say, means, variables, apeiron declaration, physics.
3. A claim names its evidence grade (`proved`, `tested`, `measured`, `model`/`anchor`) and the
   command or theorem behind it. A number is quoted from a tracked receipt or marked as not re-run.
4. The prose follows `VOICE.md`. Standard terms carry their originator; project coinages appear only
   in the glossary, mapped to the standard term.
5. The book describes the executing growable `Table`. The retired fixed-shard layout is not described.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| The book builds. | tested | `mdbook build book` |
| Every Rust block compiles against the crate. | tested | `mdbook test book -L target/debug/deps` |
| Cited paths resolve. | tested | `python3 tools/check-claims.py` |
| A broken block and a removed cited path are caught. | tested | the two controls in the plan |

### Non-goals

- Re-proving or restating a theorem; the book links the declaration.
- A performance claim not backed by a tracked receipt.
- Documentation of internal tooling that does not ship.

### Trust boundary

`mdbook test` compiles the blocks; it does not check that the prose describes them. `check-claims.py`
checks that a cited path resolves, not that the sentence is true.

### Open questions

- None.

## Specification — Public book

- **Owner:** lead
- **Intent:** `docs/epics/book.md`
- **Source revision:** `ce06316`
- **Status:** approved

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | publish a book under `book/` |
| Domain | the crate's public API, the proof pipeline in `proof/`, the engine examples |
| Totality | not applicable |
| Preconditions | `cargo build --lib` has produced `libapeiron` in `target/debug/deps` |
| Postcondition | the gate command exits 0 |

### Parts

| Part | Content | Replaces |
|---|---|---|
| Getting started | install, a `Cell`, a `Table`, a GGUF model on CPU, the Vulkan feature | `README.md` usage text |
| Theory | `R[x]/(x²−σ)`, the PDE classification, norm form, units, zero divisors, null basis, `ℤ/2⁶⁴`, `Dyadic`, addressing per face, flat/spill table, refusal, HashDoS | `docs/direction.md`, `docs/kernel-boundary.md` |
| Verification | extraction, refinement, axiom cone, evidence grades, mutation controls, trust boundary, the song correspondence | `proof/README.md` |
| Engine | GGUF and quant formats, CPU kernels, SPIR-V, measured results with commands | `docs/hash-comparison.md` for the comparison method |
| Reference | glossary (project term to standard term with attribution), bibliography | none |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| Rust block does not compile | `mdbook test` exits non-zero | block not marked `ignore`/`text` |
| Cited path removed | `check-claims.py` exits non-zero | citation names a path |

### Evidence plan

1. `mdbook build book`, `mdbook test book -L target/debug/deps`, `python3 tools/check-claims.py`.
2. Control 1: in a throwaway copy, add a block calling a nonexistent method; `mdbook test` is red.
3. Control 2: in a throwaway copy, remove a cited path; `check-claims.py` is red.

### Trust boundary

As in the intent.

### Open questions

- None.

## Plan — Public book

- **Owner:** lead
- **Intent:** `docs/epics/book.md`
- **Spec:** `docs/epics/book.md`
- **Source revision:** `ce06316`
- **Status:** approved

### Goal and completion condition

Every task gate green with its control run, then:

```sh
mdbook build book && mdbook test book -L target/debug/deps && python3 tools/check-claims.py && python3 tools/book-check.py
```

### Non-goals

- Changing a crate, proof or tool.

### Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | scaffold, introduction, install | `book/book.toml`, `book/src/SUMMARY.md`, `book/src/introduction.md` | lead | `mdbook build book` | missing SUMMARY entry target fails the build |
| W1 | cell, table | `book/src/start/cell.md`, `book/src/start/table.md`, `book/src/start/install.md` | lead | `mdbook test book -L target/debug/deps` | nonexistent method fails |
| W1 | model, Vulkan | `book/src/start/model.md`, `book/src/start/vulkan.md` | lead | `mdbook build book` | missing target fails |
| W2 | algebra, faces, scalars | `book/src/theory/algebra.md`, `book/src/theory/faces.md`, `book/src/theory/scalars.md` | lead | `mdbook test book -L target/debug/deps` | nonexistent method fails |
| W2 | addressing, table | `book/src/theory/addressing.md`, `book/src/theory/table.md` | lead | `mdbook test book -L target/debug/deps` | nonexistent method fails |
| W3 | evidence, extraction, refinement | `book/src/verification/evidence.md`, `book/src/verification/extraction.md`, `book/src/verification/refinement.md` | lead | `python3 tools/check-claims.py` | removed cited path fails |
| W3 | controls, trust boundary, song | `book/src/verification/controls.md`, `book/src/verification/trust.md`, `book/src/verification/song.md` | lead | `python3 tools/check-claims.py` | removed cited path fails |
| W4 | formats, CPU, SPIR-V, results | `book/src/engine/formats.md`, `book/src/engine/cpu.md`, `book/src/engine/spirv.md` | lead | `python3 tools/check-claims.py` | removed cited path fails |
| W4 | results | `book/src/engine/results.md` | lead | `python3 tools/check-claims.py` | removed cited path fails |
| W5 | glossary, bibliography | `book/src/reference/glossary.md`, `book/src/reference/bibliography.md` | lead | `python3 tools/check-claims.py` | removed cited path fails |

### Barrier checklist

- [x] Every task gate exit 0, with its control run and restored (`python3 tools/check-claims.py --self-test` 18/18, `python3 tools/book-check.py --self-test`, a mutated doctest fails `mdbook test`).
- [x] `mdbook build book`, `mdbook test book -L target/debug/deps`, `python3 tools/check-claims.py`, `python3 tools/book-check.py` exit 0.
- [x] Every equation carries a glossary (`python3 tools/book-check.py`).
- [x] Every number is reproduced by a doctest or carries its command; the Results chapter quotes none.
- [x] Trust-boundary statement present (`book/src/verification/trust.md`).

### Stop and rollback conditions

- A block that cannot compile against the public API: report the API gap, mark the block `text`.
- A number with no tracked receipt: omit it.
