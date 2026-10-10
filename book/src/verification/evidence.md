# Evidence and grades

Every result in this book carries one grade. The grade is part of the claim.

| grade | means | artifact that decides it |
|---|---|---|
| `proved` | A Lean kernel check passes for a named theorem. If the theorem is about Rust, the extraction is from the real source and a refinement theorem links the extracted body to the canonical definition, under a stated axiom cone. | `lake build` in `proof/` and `proof/corpus/` |
| `tested` | A `cargo test` or a doctest executes the build path and its boundary cases. | `cargo test --workspace --all-features --locked`; `mdbook test book` |
| `measured` | A benchmark ran, with a named machine, metric and interval, averaged over repeated runs. | the receipt named beside the number |
| `model` / `anchor` | A reading of the algebra onto a domain, or a design assumption. True as a mapping, not evidence of correctness. | none; labelled as such |

A citation to a theorem, a paper or another repository is a pointer. It becomes `proved` only when
the named theorem exists and the link to the Rust source is checked. The glossary under each
displayed equation carries an **apeiron:** line. That line names a declaration, and
`tools/book-check.py` fails the build of this book if the name does not resolve in the corpus.

## What a refinement theorem says

A Rust body computes on machine words. The canonical definition is a statement about a ring. A
refinement theorem connects them by a denotation: a map from words to ring elements such that the
word operation, followed by the map, equals the ring operation applied to the mapped inputs.

$$\mathrm{denote}\big(\mathrm{wrapping\_mul}(x, y)\big) = \mathrm{denote}(x)\cdot\mathrm{denote}(y)$$

- **say:** multiplying words and then reading them as ring elements equals multiplying the ring elements.
- **means:** the word multiply is the ring multiply in `Z/2^64`, with no reduced domain and no overflow case.
- **vars:** `x, y` are 64-bit words; `denote` reads the bit pattern as an unsigned number modulo `2^64`.
- **apeiron:** `word_mul_denote`, the cell-level statement built on the word-level lemma.
- **physics:** none.

The same fact can be checked by hand on the types. Reading a bit pattern as unsigned, a signed
product and the unsigned product modulo `2^64` agree.

```rust
extern crate apeiron;

let x: i64 = -7;
let y: i64 = i64::MAX;
let signed = x.wrapping_mul(y) as u64;
let unsigned = ((x as u64 as u128 * y as u64 as u128) % (1u128 << 64)) as u64;
assert_eq!(signed, unsigned);
```

## What is established and what is open

| statement | grade | where |
|---|---|---|
| `Cell<i64>` multiply and norm form refine the canonical ring operations, for every `σ` and every input | proved | `word_mul_denote`, `word_norm_denote` in `proof/corpus/WordRefinement.lean` |
| norm-form multiplicativity and the conjugate identity transfer to words | proved | `word_norm_mul`, `word_mul_conj` |
| the carrier point density and flux, the scalar carrier step, the three-cell line, the exchange kernels | proved, with stated refusal branches | `proof/README.md` lists each theorem; `proof/corpus/AxiomAudit.lean` audits them |
| the table contract on the historical fixed-shard layout | proved, revision-bound | `proof/corpus/TableRefinement.lean` |
| the executed `freq::Table` body refines the canonical `Apeiron.Table` | open | the executed body is the refinement target; see [Refinement](refinement.md) |
| the HashDoS bound | tested, not proved | `cargo test --test hashdos --locked` |
| speed against `hashbrown` and llama.cpp | measured | `cargo bench --locked --bench map`, and the commands in [Results](../engine/results.md) |

## Re-running

```sh
cargo test --workspace --all-features --locked
proof/regen.sh --check
(cd proof/corpus && lake build)
python3 tools/check-claims.py
```

`proof/regen.sh --check` re-extracts the Rust source and fails on any difference from the
committed generated file. `lake build` in `proof/corpus` includes `AxiomAudit.lean`, which throws
on any axiom outside `propext`, `Classical.choice` and `Quot.sound`. A number in this book that
comes from a document and not from one of these commands is a hypothesis until the command is
re-run.
