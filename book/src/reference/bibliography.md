# Bibliography

A citation is a pointer until the source is retrieved and read. The status column says what was
done for this book.

| source | used for | status |
|---|---|---|
| Ho, S. and Protzenko, J. *Aeneas: Rust Verification by Functional Translation.* Proc. ACM Program. Lang. 6 (ICFP), 2022. doi:10.1145/3547647, arXiv:2206.07185 | Rust to Lean extraction | abstract page retrieved; paper not read in full |
| Ho, S., Boisseau, G., Franceschino, L., Prak, Y., Fromherz, A., Protzenko, J. *Charon: An Analysis Framework for Rust.* CAV 2025 (LNCS), doi:10.1007/978-3-031-98685-7_18; arXiv:2410.18042, 2024 | front end of the extraction | abstract page retrieved; paper not read in full |
| Gauss, C. F. *Disquisitiones Arithmeticae.* Leipzig, 1801 (book, no DOI) | norm form | pointer |
| Cockle, J. "On certain functions resembling quaternions, and on a new imaginary in algebra." *Philosophical Magazine* 33, 1848, 435–439. doi:10.1080/14786444808646139 | the hyperbolic face | pointer |
| Clifford, W. K. "Preliminary Sketch of Biquaternions." *Proc. London Math. Soc.* s1-4, 1873, 381–395. doi:10.1112/plms/s1-4.1.381 | the parabolic face | pointer |
| Knuth, D. E. *The Art of Computer Programming*, vol. 3, section 6.4 (book, no DOI; ISBN 978-0-201-89685-5) | multiplicative hashing | pointer |
| de Moura, L. and Ullrich, S. *The Lean 4 Theorem Prover and Programming Language.* CADE-28 (LNCS), 2021. doi:10.1007/978-3-030-79876-5_37. The Mathlib Community. *The Lean Mathematical Library.* CPP 2020. doi:10.1145/3372885.3373824. Pinned at `v4.31.0` | the proof corpus | the pinned toolchain is built and run by `proof/regen.sh` and `lake build` |
| ggml and llama.cpp (github.com/ggml-org/llama.cpp; no DOI) | GGUF and the quantisation block layouts; the throughput comparator | the repository pins the comparator revision in each receipt |

Each DOI above was resolved against Crossref's record for the cited title, authors and year; no DOI means the source has none. Retrieving and reading the pointer entries is open work. Until then each is a pointer and carries
no evidential weight in the [grades](../verification/evidence.md).
