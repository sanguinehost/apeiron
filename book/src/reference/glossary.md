# Glossary

Each project term is mapped to the standard term. Standard names are used throughout the book; a
project coinage appears here only so that source comments can be read.

| in the source | standard term | originator, first reference |
|---|---|---|
| `Cell<R>` | element of the quotient ring `R[x]/(x^2 - sigma)` | Cockle (1848), split-complex; Clifford (1873), dual numbers |
| `Sigma` | the parameter `sigma` of the quotient; the sign of `sigma` classifies the face | |
| `Elliptic`, `Parabolic`, `Hyperbolic` | `sigma < 0`, `sigma = 0`, `sigma > 0` | the names are those of the discriminant classification of second-order linear PDEs; the reading of the algebra onto that classification is an `anchor` |
| `norm_form` | norm form `a^2 - sigma b^2` | Gauss, *Disquisitiones Arithmeticae* (1801) |
| `conj` | conjugation `a + bx -> a - bx` | |
| idempotent | `e` with `e^2 = e` | |
| zero divisor | non-zero `u`, `v` with `uv = 0` | |
| `Coord` | an injective map from a key to a point of the ring | |
| `freq::Table` | open-addressing hash table with Fibonacci multiplicative hashing | Knuth, TAOCP vol. 3 (1973), section 6.4 |
| `PHI` | the odd 64-bit golden-ratio multiplier | Knuth, TAOCP vol. 3 |
| `PROBE_CAP` | bound on the linear probe length before overflow | |
| overflow | the out-of-line store for keys that exceed the probe cap | |
| Q32.32 | signed fixed-point with 32 fractional bits | |
| `Q8_0`, `Q4_K`, ... | ggml block quantisation formats | llama.cpp / ggml project |
| carrier | the packed integer width (1, 2, 4 or 8 bit) of a weight | |
| `Halve` | a ring where `2` is invertible | |
| `Dyadic` | the rational `m / 2^e` | |
| extraction | translation of Rust to Lean by Charon and Aeneas | Ho and Protzenko (2022), Aeneas; Ho et al. (2024), Charon |
| refinement | a proved relation between the extracted body and a canonical definition | |
| axiom cone | the axioms a Lean theorem depends on | |
| `anchor` | a design assumption or a reading of the algebra onto a domain | |

## Grades

The grades are defined in [Evidence and grades](../verification/evidence.md): `proved`, `tested`,
`measured`, `model` and `anchor`.
