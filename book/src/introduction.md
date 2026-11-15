# Introduction

apeiron is a Rust crate and a Lean development. The crate holds one algebra, `R[x]/(x² − σ)`, and a
hash table built on it. The Lean development states what the algebra and the table mean and checks
that the Rust code means it.

The deployment goal is a 7B-class model on the 6.6 GiB laptop: about 30 tok/s on the CPU, a
200,000-token context, and at least 2 GiB left for four or five coding-agent sessions. The model is
served through `serve/` into `../intent`, and the quantiser, the context method and the Vulkan/SPIR-V
kernels are open source, with the kernels verified through Aeneas and Lean. [The goal](goal.md)
states the acceptance criteria. `GOAL.md` at the repository root is the canonical statement of work,
status and non-goals. A pack, a benchmark row or a speedup is an artifact of the goal, not the goal.

The book has five parts.

| Part | Reader | Content |
|---|---|---|
| Getting started | anyone who builds Rust | install, a `Cell`, a `Table`, a model on the CPU and GPU |
| Theory | anyone who wants the reason | the algebra, its three faces, the table |
| Verification | anyone who wants to check | what is proved, tested and measured, and what is not covered |
| Engine | anyone who runs a model | formats, the integer CPU path, SPIR-V, how throughput is measured |
| Reference | anyone | glossary and bibliography |
