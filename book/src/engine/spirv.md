# SPIR-V and Vulkan

The GPU path builds SPIR-V modules in Rust and dispatches them through a hand-written Vulkan
runtime. The runtime loads `libvulkan.so.1` with `dlopen` and depends on no crate. It is behind the
`gpu` feature (see [Run on the GPU](../start/vulkan.md)).

| layer | location | role |
|---|---|---|
| module builders | `src/simt/` | `_bytes` functions that emit SPIR-V words: GEMV, DP4A GEMV, batch GEMM, reductions, elementwise ops |
| runtime | `src/gpu/` | device, buffer, pipeline and dispatch; `Gpu::upload`, `pipeline`, `bind`, `run` |
| whole-model decode | `src/gpu/decode/` | `GpuDecode`: model upload, per-layer step, prefill |
| fragment checker | `src/spirv/` | `check_fragment`: a pure check of an emitted module |
| shaders | `src/spirv/*.comp`, `*.spv` | cooperative-matrix GEMM and attention, split-K reduce |

The builders stay compiled without the `gpu` feature, so a module can be generated and checked on a
machine with no Vulkan device.

## The fragment checker

`check_fragment` decodes a SPIR-V 1.3 module and applies these obligations before the module is
used:

| id | obligation |
|---|---|
| header | the five-word header |
| D1 | `WordCount` framing |
| D2 | result-id uniqueness and definition before use |
| D3 | `OpSUDot` integer type agreement |
| D4 | admitted storage classes and decorations |
| opcodes | membership in the served opcode set |

It performs no float semantics: float epilogue opcodes are admitted as syntax and checked for
binary shape only. It never panics on input.

## Grade

| claim | grade | artifact |
|---|---|---|
| the checker accepts and refuses the cases in its tests | tested | `cargo test --workspace --all-features --locked` |
| the checker body is extracted from the real source (`crate::word_check_fragment` in `proof/extract/lib.rs`) | extraction | `proof/regen.sh --check` |
| the header check refines its Lean statement; component lemmas cover the scan, the tables and the `OpSUDot` checks | proved for the header | `Spirv.checkFragment_header` |
| whole-checker refinement of D1 to D4 and the opcode set | open | none |
| an accepted module computes the intended function | not claimed | none |

The files are `proof/corpus/Spirv/Check.lean` and its `CheckScan`, `CheckScanState`, `CheckApplyShape` and `CheckApplyShapeAssembly` companions. Each states its own residual obligation in its header.

## Carrier lemmas

`Spirv.carrier_split` in `proof/corpus/Spirv/Carrier.lean` proves the integer carrier split that the
DP4A kernels use, with `Spirv.carrier_decode` and `Spirv.carrier_split_cells`. All three are audited in
`AxiomAudit.lean`. The Lean statements read the emitter's unpack lines as an `eval` denotation; the
lines are matched by quotation, not by extraction. `Spirv.carrier_decode` covers widths 1, 2, 4 and 8
only.

## Not covered

The GPU driver, the shader compiler inside it and the hardware are outside every claim here. The
GPU path is single-tenant: buffers run without `robustBufferAccess` and workgroup memory is not
cleared between dispatches.
