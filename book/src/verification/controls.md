# Controls

A check that cannot fail has no weight. Each gate in this repository ships a control: a deliberate
defect, applied to a throwaway copy, that must turn the gate red. Restoring the copy must turn it
green. A gate is accepted only after its refusal has been seen under the harness that runs it.

## Proof controls

| defect applied to a throwaway copy | gate | observed red |
|---|---|---|
| dial value for the hyperbolic face changed from `1` to `2` in `src/cell.rs`, then re-extract | `WordRefinement` | `sigma_value_eq` and the four multiplication and norm theorems |
| derived impulse changed from `p.a.sub(q.a)` to `p.a.add(q.a)` in `src/exchange.rs`, then re-extract | `WordRefinement` | `word_exchange_impulse_denote`, `word_elastic_exchange_denote` |
| `then` and `else` arms of the acceptance branch swapped in the generated file | `WordRefinement` | `word_accept_derived_impulse_accepts` and two more |
| fresh-write merge argument order swapped in the generated table call | `TableRefinement` | `table_insert_empty_zero` |
| the final `ok (some …)` of the three-cell line replaced by `ok none` | `WordRefinement` | `word_diffusive_line3_step_zero_success` |

The last row matters most. It shows the success witness cannot be satisfied by an implementation
that always refuses.

## Test controls

`cargo test` covers the nominal path and the boundary cases the contract names. A source mutation
in a throwaway copy must fail a test.

## Book controls

The book is a gated artefact.

| defect applied to a throwaway copy | gate |
|---|---|
| a code block calls a method that does not exist | `mdbook test book` |
| a glossary line names a declaration that does not exist | `python3 tools/book-check.py` |
| a cited path is removed | `python3 tools/check-claims.py` |

`python3 tools/book-check.py --self-test` runs the second control against built-in cases.

## Wall-clock budget

A local gate invocation declares a numeric wall-clock budget and is enforced across its
descendants. A timeout is neither a counterexample nor a successful control. A gate that cannot
finish within its budget stays blocked and is redesigned without weakening its contract.
