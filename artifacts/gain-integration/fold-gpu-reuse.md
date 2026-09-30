# Fold GPU reuse decision

Decision: start with Fold's existing SPIR-V kernels, typed kernel selection and resident runner.
GPU prefill is an integration/refinement task, not a new backend or dot-kernel construction task.
This is a source architecture review (`anchor`/planning); no build, benchmark or admission ran.
Historical near-CUDA comments are source pointers, not fresh speed evidence.

## Identity and ownership

| tree | read identity | scope |
|---|---|---|
| Apeiron | `01de89d020954897f48e9863f806ac9df0bbf5af`, dirty supplied checkout | report only; parent owns epic edits |
| Fold | `d497161f32e0d366220f2065ec36e680bd9aed86`, dirty supplied checkout | read-only migration source; existing dirty files preserved |

Documentation exception: `APEIRON-GPU-PREFILL-SPIKE-fold-reuse` writes this report in the supplied
checkout. No build lane, branch, worktree, shared-cache mutation, commit or external post.
Time budget: 180 seconds for this read-only slice; the report's path/digest gate is bounded at 10 s.

```text
SCOPE:      artifacts/gain-integration/fold-gpu-reuse.md; reuse interfaces and next bounded task.
REGISTER:   planning; implementation, proof acceptance, benchmarks and admission excluded.
CORPUS:     Spirv.gpu_dp4a_word_refines_cpu_word, under face = realizedSigma:
            Refines (gpu_dp4a_word M K segs X W r y) (cpu_dp4a_word M K segs X W r y).
REFERENCE:  /home/socol/Workspace/fold/bedrock/src/serve/gpu.rs;
            /home/socol/Workspace/fold/modelfold/src/simt/superblock.rs.
GATE:       report-presence plus primary-path/digest check, cwd /home/socol/Workspace/apeiron;
            baseline test -f artifacts/gain-integration/fold-gpu-reuse.md exits 1, absent report.
GRADE:      anchor/planning; no fresh GPU proof, test or performance acceptance.
```

## Existing reusable surfaces

Every Fold path below is relative to `/home/socol/Workspace/fold/` and was inspected at the identity above.

| primitive | exact existing API / source | reusable contract |
|---|---|---|
| Quantized decode projection | `modelfold/src/simt/dp4a.rs:45`, `gemv_dp4a_module_dyn(m,k,local,segments,carrier,dyn_sx)`; `_dyn_bytes_sc` at 921 | Q8/Q4/Q2 chart, dynamic block scales, split-K; integer dot followed by f32 scale/bias accumulation |
| Batched projection | `dp4a.rs:960`, `gemv_dp4a_batch_module(m,k,local,carrier,bt)` | Same decode chart, no repack; one weight load reused across `bt` activation rows |
| Register-tiled prefill projection | `modelfold/src/simt/superblock.rs:55`, `gemv_dp4a_batch_rt_module(m,k,local,carrier,bt,rt)` | Reuses activation loads across `rt` output rows and weights across `bt` tokens; unrolled inner block; `WorkgroupId.y` batch addressing |
| Superblock projection | `superblock.rs:850`, `gemv_dp4a_sb_module_ex`; q2/q4 chart transpose helpers at 1812/1853 | Distinct per-256 scale layout and optional split-K/LDS/physical-address forms; retain layout choice explicitly |
| Kernel configuration | `modelfold/src/simt_nn/spec.rs:316`, `GemvSpec { carrier,scale,mapping,fusion,reduce,experts }` | `legalize(k) -> Result<(),SpecError>`, `chart_bytes(m,k)`, `groups(m,local)`, `emit(m,k,local) -> Result<Vec<u8>,SpecError>`; typed scale/mapping/fusion/reduction rather than flags |
| Batch glue | `modelfold/src/simt_nn/batch.rs:367/451/490/577/734/787` | Existing batch RMSNorm+quantize, plain quantize, superblock variants, SwiGLU and QKV+RoPE; `last_row_gather_module`/`_sb_module` avoid whole logit slabs |
| Prefill attention | `modelfold/src/simt_nn/prefill.rs:27/277`, `attn_prefill_module`, `attn_prefill_tiled_module(...,rows,rts,ct,fuse)` | Existing causal dense attention and tiled workgroup variant; shape constraints explicit; fusion changes float association |
| Hybrid recurrence | `modelfold/src/simt_nn/deltanet.rs:48/184/292/404/452` | Delta rule, convolution+SiLU, L2 normalization, decay/beta, gated normalization already emitted and used in decode |
| Low-level runtime | `modelfold/src/gpu/runtime.rs:71`, `Gpu` | Persistent mapped staging and command buffer; device/host-read allocation, upload/write/scatter, pipeline/bind, `run`, `time`, `staged_dispatch` |

The integer core is `OpSUDot`; emitted prefill outputs remain f32. For example,
`superblock.rs:335-340` computes `d * int_dot * sx + lo * xs`, then adds to its running f32.
Replacing that epilogue with integer fixed point changes the observable contract; it is not reuse
of the existing served arithmetic. `gpu.rs:4059` explicitly distinguishes GPU association from
CPU block-order association and names max absolute error/ULP/argmax observations.

## Layout and launch interface

Reuse the six binding ABI of the per-32 batch/row-tiled kernel (`dp4a.rs:942-960`):

| binding | element/address | extent |
|---|---|---|
| 0 W | `u32`, `W[c*M+r]`, `pack` codes/word | `M*K/pack` |
| 1 XQ | `u32`, token `i` starts at `i*K/4` | `Bp*K/4` |
| 2 OUT | `f32`, `i*M+r` | `Bp*M` |
| 3 DM | `u32`, f16 `(d,lo)` at `b*M+r` | `M*K/32` |
| 4 XS | `f32`, block input sum `i*(K/32)+b` | `Bp*K/32` |
| 5 SX | `f32`, block input scale, same index | `Bp*K/32` |

Domain: `K % 32 == 0`, chart carrier, positive tile dimensions, positive M for the `M-1`
clamp, validated device local size. Pad token storage to `Bp = ceil(B/bt)*bt`; the kernel has no
batch guard. Ordinary batch groups are `(ceil(M/local),Bp/bt)`; row-tiled groups are
`(ceil(M/(local*rt)),Bp/bt)`. Row-tiled loads clamp row to `M-1`, while final stores retain the
live-row predicate (`superblock.rs:158-173/351-369`). A theorem for guarded untiled loads does not
automatically cover this clamped row-tile body. Checked size arithmetic and allocation outcomes
must be part of the Apeiron adapter, not inferred from these emitter assertions.

## Resident execution already exists

| existing runner | state/ownership/observations | actual scope |
|---|---|---|
| `bedrock/src/serve/gpu.rs:974`, `GpuDecode::new(model,tmax)`, `step(&mut self,pos,tok)` | Owns Vulkan context and replayed command buffer; persistent weights/intermediates/KV; `owner`, `dev_len`, `seed_kv`; recurrent buffers retained and `reset_recurrent` at 3566; `n_disp/n_bar`, submit/read/seed counters | Full decode, including hybrid DeltaNet dispatches at 2404-2461 |
| `gpu.rs:4072`, `GpuPrefill::new(model,maxb)`, `matmat(&mut self,&Proj,&[&[f64]],&mut[f32]) -> bool` at 4333 | Uploaded chart cache keyed by `Proj::words.as_ptr`, reusable pipelines/descriptors/scratch; capacities and scope select fallback; quantize/upload/readback counters | Projection-only offload: CPU glue remains and every projection transfers activations/results |
| `gpu.rs:4573`, `GpuPrefillStack::new(model,maxb)`, `run` at 5427 | Uploaded charts/gains, persistent scratch/KV and planned dispatch nodes; one submit/drain per chunk; KV mirror seed only for changed owner or missing prefix; `invalidate`, `LogitRows::{All,Last}`, borrowed `logits`/`last_logits` | Whole dense prefill stack: norm1→QKV→RoPE/KV→attention→quantize→O→norm2→gateup→SwiGLU→quantize→down; final norm and selected head tail |

The resident stack is the primary integration reference. Its constructor checks uniform fused QKV,
head width and chart support (`gpu.rs:4711-4750`); this excludes the hybrid shape rather than
providing a batched DeltaNet path. `run` has explicit refusal/fallback for empty/oversized chunks,
sparse/instrumented reads, changed KV capacity, out-of-window positions and invalid logit output
shape (`5427-5491`). It commits KV mirrors for CPU continuation; last-row mode avoids computing
all head rows. Reuse those semantics and counters instead of introducing a fresh resident graph.

Ownership requires work: runner submission uses exclusive `&mut self` and `Send`, not `Sync`;
mapped result borrows end before the next run. The runtime's `Buffer` is `Copy` and deliberately
has no `Drop` (`runtime.rs:26-38`); handles currently leak. This is existing trusted lifecycle
behaviour, not a proved allocation/free ownership API. An adapter should specify context lifetime,
buffer lifetime and reclamation explicitly, without silently inventing reclamation guarantees.

## Proof correspondence and bounded extensions

Apeiron `docs/epics/gpu-spv.md`, `src/spirv/check.rs` and `proof/corpus/Spirv/` already own the
barrier-free integer seam. `tests/gpu_spv_check.rs` captures Fold's dynamic Q4 decode module from
the same Fold revision and DP4A source hash. This is a useful concrete starting fixture.

| current artifact | exact existing content | next link |
|---|---|---|
| `Spirv/Semantics.lean:171/592` | `sdot_denote`, `sdot_split_ringMul`; `realizedSigma = 1`, channel-0 regular-representation reading | Reuse for each actual packed integer contribution; retain signed/unsigned lane orientation and missing cross-channel result |
| `Spirv/Dp4a.lean:220/234/252` | Model GPU/CPU words refine `dp4a_gemv_spec`, fixed-point scale law | Actual CPU body and decoded emitted module links remain absent; theorem does not describe existing f32 epilogue |
| `Spirv/Dp4a.lean:365` | Guarded model access-chain bounds | Extend to actual row-clamp/store predicate and padded batch ABI, then constructor-derived capacities |
| `Spirv/Check.lean:45-65` | Header bridge plus explicit framing/shape, scan-state and composition residuals | Close named components in existing GPU-SPV work; do not duplicate checker ownership |

No fresh Lean gate ran, so this review quotes existing theorem statements rather than asserting
fresh `proved` acceptance. Whole-SPIR-V proof is unnecessary for a reuse design. Individual selected
actual-body links, graph/state observations and admitted source/cost contracts still gate implementation.

Capabilities are already partitionable: row-tiled per-32 DP4A is barrier-free; batch norms, attention
and DeltaNet use `Workgroup` memory/`OpControlBarrier` (`simt_nn/mod.rs:736-759`); `coop.rs` adds
subgroup arithmetic; `simt/gemv.rs:1073` supplies a separate f16 cooperative-matrix GEMM with
Float16/16-bit storage/Vulkan memory-model/cooperative-matrix capabilities. These are reusable
tested-runtime candidates with separate proof seams, not reasons to rewrite the existing DP4A path.
Keep physical addresses and saturating-dot variants separate too. Reject unsupported opcodes/capabilities
under the selected proof profile; do not widen the v0 checker silently.

## Existing validation and missing fresh evidence

- `modelfold/tests/simt.rs` validates emitted modules and corrupted-module rejection;
  `modelfold/tests/simt_nn.rs` validates glue. Both can return a successful test after an explicit
  missing-validator skip, so a reused gate must require a validator-presence receipt.
- `modelfold/examples/simt_nn_check.rs:130` executes embed, norm, SiLU/add, QK norm and dynamic DP4A;
  no Vulkan yields SKIP. It does not establish whole resident/hybrid prefill equivalence.
- `bedrock/examples/qwen35_gpu_gate.rs` compares the existing hybrid decode chain with CPU forward;
  useful recurrent state/head mapping controls, distinct from batched prefill.
- `bedrock/src/bin/servefold/prefill.rs` is the existing real-token prefill instrument;
  `bedrock/examples/simt_profile.rs` profiles decode kernels/driver IR/energy. Neither has been run
  here. Historical bandwidth/near-cuBLAS comments remain unrefreshed.

## Next bounded task and epic replacement

Replace the fresh resident-dispatch/kernel task with a source-bound adapter design:
`APEIRON-GPU-PREFILL-W1-T2`, owner Luna, independent Sol review, at most three paths:
this reuse report, `docs/epics/gpu-prefill.md`, and one new adapter-contract artifact. Parent owns
the current epic edit until handback; no overlapping writer. No inference implementation changes.

Choose Fold's six-binding register-tiled Q8 per-32 projection and existing dense resident-stack
schedule as the first interface. Inventory Apeiron's actual Q8 packing/float carrier against Fold's
W/DM/XQ/XS/SX ABI; specify a lossless conversion or a named incompatibility. Specify stable model
and session identity (rather than persisting raw pointer identity), constructor/refusal semantics,
KV and recurrent state conversion, immutable kernel hashes, padding/live ranges and exact outputs.
Reuse existing norm/attention/DeltaNet modules as individually registered components; hybrid
batched recurrence scheduling is the additional graph task after dense integration.

The adapter contract must select integer-core exact observations plus a separately authorized f32
epilogue/association policy; it cannot inherit CPU exact-output equality from an argmax comparison.
Select the existing untiled/tiled controls for wrong lane signedness, packing stride, padded token
extent, row-store predicate and session carry reset. Write the concrete red source/layout gate
before allocating its implementation task; that gate is currently MISSING. Bind resource counts
(uploads, bytes, dispatches/drains, arithmetic, peak-live storage) to the already executing graph
for `resource-guided-evolution.md`, rather than inventing a second graph/cost interface.

Then refresh matched whole-prefill measurements on the existing Fold kernels with named
models/shapes/carriers and cold/resident policies; isolated profile timings cannot be added to
produce a whole-graph time. No optimization worker starts until actual source/refinement/cost
registration, independent verdict and `tools/dispatch/admit.py --launch` accept its bound argv.

## Source digests

| Fold source | sha256 |
|---|---|
| `modelfold/src/simt/dp4a.rs` | `ffe4f6c313492b84ecfc2c5d99c9fa8b2945f5bf081282e9b13b1374da50901e` |
| `modelfold/src/simt/superblock.rs` | `e15e18bacfa02b5508dfefc959fc9bd7f2ef1593ff208d8de6e5e3c716745387` |
| `modelfold/src/simt_nn/prefill.rs` | `0d5da9a491d5223d1f8f9cf6b8d02a64d23b373bb2cef74384845b09c3e54d75` |
| `modelfold/src/simt_nn/batch.rs` | `d183f9cb8decb9ffccb37c91a7973c39003b33361babc8dc6f98615cfc618fd7` |
| `modelfold/src/simt_nn/deltanet.rs` | `c72f1859e82a974339b208ccf6ae67455fa7be73b85f6845887c5d6b69906879` |
| `modelfold/src/simt_nn/spec.rs` | `f54e9fe3590aa4c6c87d382a8a4c9b76a56376eda58d212e4f2dd725a2c445b0` |
| `bedrock/src/serve/gpu.rs` | `1e01838dca24f7751274249af671bb485109b80f8f0e7f9fa940e73744e9b17c` |

The trust boundary includes raw Vulkan FFI, unreclaimed resource handles, shader/toolchain lowering,
driver/device execution and host measurement provenance. None is discharged by the word algebra.

Report gate: `timeout 10s python3` checked report presence, all seven live Fold source digests and
required reuse/boundary fields: exit 0, internal elapsed 0.001 s. Baseline report-presence command
exits 1 before creation. No inference, Rust, Lean, device, benchmark or admission gate ran.
