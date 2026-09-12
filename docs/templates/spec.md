# Specification — <epic title>

- **Owner:** <name>
- **Intent:** `docs/intent/<slug>.md`
- **Source revision:** <git rev-parse HEAD>; working tree <clean|dirty, list files>
- **Status:** draft | approved

## Scope and admissible domain

<What operation is specified. State the domain explicitly.>

| Item | Value |
|---|---|
| Operation | <Rust symbol or Lean declaration> |
| Domain | <every admitted input, with width/type/range> |
| Totality | total / `Option` / `Result` / checked / panics |
| Preconditions | <named, or "none"> |
| Postcondition | <the exact relation the result satisfies> |

No silent domain reduction. If a proof needs a narrower domain, that domain is named here and the
operation keeps its honest partial signature.

## Representation relation (Rust to Lean)

<The exact link. Name the real source, the extracted body, and the canonical definition it refines.>

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `<fn>` | `src/<file>.rs` |
| Extraction wrapper | `word_<fn>` | `proof/extract/lib.rs` |
| Extracted body | `<module>.<fn>` | `proof/ApeironWord.lean` (generated) |
| Canonical definition | `<Namespace.definition>` | corpus / `proof/corpus/WordRefinement.lean` |
| Refinement theorem | `word_<fn>_denote` | `proof/corpus/WordRefinement.lean` |

The wrapper fixes a type; it does not restate the formula. The refinement theorem is a link only if
it resolves and the axiom audit passes.

## Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| <name> | <universally quantified relation> | proved | `<theorem>` |
| <name> | <behavior on boundary input> | tested | `<test fn>` |

## Failure semantics

<For each failing input, what the operation returns or raises. `None`, `Err`, panic, saturation and
the exact condition. An infallible signature over a lossy operation is a defect, not a contract.>

| Input class | Result | Named condition |
|---|---|---|
| <overflow / unrepresentable / empty> | <`None` / `Err` / panic> | <precondition or check> |

## Evidence plan

1. **Positive tests.** `cargo test --workspace --all-features --locked` covers the nominal path and
   the boundary cases above. Name the test file.
2. **Proof.** `proof/regen.sh` regenerates the extraction; `cd proof/corpus && lake build` checks the
   refinement theorem.
3. **Axiom cone.** `proof/corpus/AxiomAudit.lean` `#audit`s the headline; anything outside
   `{propext, Classical.choice, Quot.sound}` fails the build.
4. **Negative control.** <The mutation in a throwaway copy that must turn the gate red, and the
   command that shows it red. Restore and show green.>
5. **Measurement** (only for performance claims). <Command, machine, metric, before/after.>

## Trust boundary

<What this specification and its evidence do not cover.>

## Open questions

- <question and the decision needed>
