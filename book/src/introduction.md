# Introduction

apeiron is a Rust crate and a Lean development. The crate holds one algebra, `R[x]/(x² − σ)`, and a
hash table built on it. The Lean development states what the algebra and the table mean and checks
that the Rust code means it.

The book has five parts.

| Part | Reader | Content |
|---|---|---|
| Getting started | anyone who builds Rust | install, a `Cell`, a `Table`, a model on the CPU and GPU |
| Theory | anyone who wants the reason | the algebra, its three faces, the table |
| Verification | anyone who wants to check | what is proved, tested and measured, and what is not covered |
| Engine | anyone who runs a model | formats, the integer CPU path, SPIR-V, how throughput is measured |
| Reference | anyone | glossary and bibliography |
