# Introduction

`apeiron` is a Rust crate and a Lean 4 development built around one algebra, `R[x]/(x² − σ)`. The
crate implements that algebra (`Cell<R>` and its norm form `a² − σb²`), deterministic flat/spill
hash tables (`freq::Table`, `CoordMap`, `FreqMap`), and a fixed-point CPU/Vulkan inference engine
for GGUF models. The Lean development in `proof/` extracts the `Cell<i64>` word bodies through
Charon and Aeneas and refines them against canonical definitions in `proof/canonical/`.

## Why these pieces share one tree

One ring with `σ ∈ {−1, 0, 1}` unifies the complex (elliptic), dual (parabolic), and split-complex
(hyperbolic) faces. Passing `Sigma` as a value keeps face-specific rules — such as two independent
channels and lossy discard on the hyperbolic face — inside the type system rather than across
separate implementations.

The CPU inference engine applies the same discipline: `matmat_q8act` is one projection loop over a
carrier dial (`F16`, `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`, and 1-bit sign factors) for both decode
(`k = 1`) and prefill (`k > 1`). Keeping the algebra, tables, quantiser, context method, and
kernels in one repository keeps the integer accounting (Q32.32 fixed-point on the token path),
overflow refusal route, and verification boundary under one set of gates.

## Goal and evidence

The deployment target is a 7B-class model on a 6.6 GiB laptop at about 30 tok/s CPU decode with a
200,000-token context and at least 2 GiB free memory for coding-agent sessions, served through
`serve/` into `../intent`, with the quantiser, context method, and Vulkan/SPIR-V kernels shipped
under Apache-2.0 and verified through Aeneas and Lean. **That goal is open.** [The goal](goal.md)
summarises the acceptance criteria; `GOAL.md` at the repository root is the canonical statement of
work, status, and non-goals.

Every claim in this book carries an evidence grade: `proved` (a Lean kernel check), `tested` (a
named Cargo test), `measured` (a named command and host), or `model`/`anchor` (a design mapping).
Nothing below the extraction — `rustc`, LLVM, the kernel ABI, Vulkan drivers, or GPU hardware — is
in the proofs, and the crate's cryptographic modules are unaudited.

## How to read this book

| Part | Reader | Content |
|---|---|---|
| [Getting started](start/install.md) | anyone building Rust | install, `Cell`, `Table`, running a model on CPU and GPU |
| [Theory](theory/algebra.md) | anyone wanting the math | `R[x]/(x² − σ)`, the three faces, scalars, addressing, the table |
| [Verification](verification/evidence.md) | anyone auditing claims | evidence grades, extraction, refinement, controls, trust boundary |
| [Engine](engine/onepath.md) | anyone running or tuning inference | one path, GGUF formats, NanoQuant packs, CPU kernels, SPIR-V, benchmark protocol |
| [Reference](reference/glossary.md) | anyone looking up terms | glossary and bibliography |

- To build the crate and run your first example, start at [Install](start/install.md).
- To run a GGUF model or NanoQuant pack, go to [Run a model](start/model.md) and
  [One path](engine/onepath.md).
- To audit what is proved, tested, and open, read [Evidence and grades](verification/evidence.md)
  and [Trust boundary](verification/trust.md).
