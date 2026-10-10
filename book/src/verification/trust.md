# Trust boundary

The evidence reaches the extracted Rust bodies and stops at the layers below.

## Not covered

| layer | why |
|---|---|
| rustc and LLVM code generation | the proof reads MIR, not machine code |
| the kernel ABI and the allocator | the floor under any userspace program |
| Vulkan drivers, SPIR-V compilers and the GPU | outside the extracted word claim |
| the `unsafe` container | the model turns a bad read into a failing proof obligation; it does not discharge it |
| the Lean model definitions for intrinsics | a definition read against the Rust and Intel semantics; wrong ones are wrong premises |
| the AVX2 detection result | modelled as an unconstrained boolean |
| thread memory semantics, panics and liveness | no model |

## Counted

| item | size and statement |
|---|---|
| Lean kernel | trusted by construction |
| Aeneas and Charon | pinned; a translation bug is a wrong premise |
| `proof/patches/aeneas-lean-model.patch` | the external model definitions, listed in [Extraction](extraction.md) |
| axiom cone | `propext`, `Classical.choice`, `Quot.sound` for the Rust bridge and the canonical corpus; the empty set for the Slang chain |
| extraction boundary axioms | four `f32` attention-carrier operations the Aeneas model does not define, listed in `proof/boundary-axioms.txt`; a theorem about their users holds relative to them |
| checking code | `tools/check-claims.py`, `tools/book-check.py`, `proof/regen.sh`, `AxiomAudit.lean` |

## Cryptography

The `aes`, `chacha`, `sha256` and `crypto` modules are checked against known-answer vectors only.
They are not side-channel resistant and have not been reviewed. Do not use them to protect real
data. See `SECURITY.md`.

## The GPU path

Vulkan buffers run without `robustBufferAccess` and workgroup memory is not cleared between
dispatches. The path is single-tenant.

## Performance claims

A number is a measurement only when a named command ran and its output was read. It names the
machine, the metric and the interval, and it is the mean of at least three runs with the observed
spread. A figure that moved within the spread already seen on unchanged code is not a change.
