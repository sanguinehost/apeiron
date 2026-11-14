# Candle and Burn review for apeiron-nq

Source read only, no builds: `huggingface/candle` at `cfcbd7186d13271843ce7eb41d25659bd0835850`
(2026-11-01), `tracel-ai/burn` at `ce8babc9a8b0f696fd5fa10ea98bf2fb17bdf4e5` (2026-11-12). Both are
comparison fixtures; neither is a dependency and neither ships. The browser layer is apeiron's own
wasm32 build (§3.3). Every claim below is a code read (`anchor`); the only numeric performance
claims are quoted from the repositories' own example READMEs and are not reproduced here. Paths are
relative to each clone.
Tags: **impl** = code implements it; **example** = wired example; **absent** = not found;
**stub** = explicit `todo!`/`unimplemented!`/doc-only.

## 1. Candle 0.11.1

### 1.1 GGUF and quantized inference

| item | status | evidence |
|---|---|---|
| GGUF v1/v2/v3 reader | impl | `candle-core/src/quantized/gguf_file.rs:52-78`, `:123-128`, `:569-580` |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | impl | `GgmlDType` at `candle-core/src/quantized/mod.rs:285-301` |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | impl | same enum; `impl GgmlType` at `k_quants.rs:819,1057,1438,1716,2021,2318` |
| F32/F16/BF16 GGUF tensors | impl | `mod.rs:286-288,321` |
| IQ2/IQ3/IQ4 etc. | **absent** | `mod.rs:304-323` has no ids 16-29; unknown id `bail!` |
| mmap for GGUF | **absent** | `gguf_file.rs:110-113` allocates `vec![0u8; size]` and `read_exact` per tensor |
| dequantize | impl | `GgmlType::to_float`, e.g. Q4_0 `k_quants.rs:207-226`; Q4_K `:1438+` |
| quantized matmul CPU | impl | `QTensor::cpu_fwd` `mod.rs:1198-1285`; `k_quants::matmul` `:2421-2670` |
| quantized matmul CUDA/Metal | impl | `mod.rs:130-132`, `:1288-1310`; `cuda.rs`, `metal.rs` |

**Q4_K_M on CPU: yes, with a memory problem.** The Qwen3 example downloads `Q4_K_M.gguf` for every
size including 8B (`candle-examples/examples/quantized-qwen3/main.rs:127-135`) via `QMatMul`
(`quantized_qwen3.rs:463`). CPU dispatch:

1. `QMatMul` keeps the tensor quantized unless the dtype is F32/F16/BF16 (`mod.rs:865-880`).
2. `cpu_fwd` first tries `repack::try_matmul_f32` (`mod.rs:1231`), which on x86_64 requires
   `n % 16 == 0`, `k % 256 == 0` and dtype Q4K/Q6K/Q8_0 (`repack_x86.rs:43-60`), repacks weights
   into tiles and runs AVX-512-VNNI or AVX-VNNI (`repack.rs:256-275`; AMX Q4K for `m >= 32`,
   `:264-269`). aarch64 uses NEON/i8mm (`repack.rs:226-253`).
3. Fallback is `matmul_t` → `k_quants::matmul`, parallelised by candle's own `barrier_pool`
   (`k_quants.rs:2490`, pool sized from `CANDLE_NUM_THREADS`/`RAYON_NUM_THREADS`,
   `utils.rs:411-421`). Rayon is used elsewhere, not in this kernel.
4. F16 lhs goes through a scalar loop (`k_quants.rs:2678-2710`), so use F32 activations.

**Memory arithmetic for Qwen3-8B Q4_K_M (not measured).** File 5.03 GB (apeiron bench log); the
embedding is dequantized to F32 (`quantized_qwen3.rs:492-493`): 151936 × 4096 × 4 = 2.49 GB. The
x86 tile form stores F32 scales/mins per 16 rows, ≈6 vs 4.5 bits/weight (`repack_x86.rs:14-21`),
adding ≈1.3× on the quantized part. Total ≈8.5 GB before KV cache: 8B does not fit 6.7 GB, 4B
(≈4.4 GB) does. No switch disables repacking; `CANDLE_DEQUANTIZE_ALL` (`mod.rs:843-860`) is worse.

### 1.2 Extension points

| surface | what exists | cost for a custom NQ linear |
|---|---|---|
| `CustomOp1/2/3` | trait with `cpu_fwd`, optional `cuda_fwd`/`metal_fwd`, and optional `bwd` (`candle-core/src/custom_op.rs:8-41`); applied with `Tensor::apply_op1` (`:192`), no-bwd variant `:157` | CPU-only op is pure Rust in the app crate, no build-system change; ~100-300 LOC |
| existing template | `QTensor` itself is a `CustomOp1` holding weights out-of-band (`mod.rs:1198-1310`) | this is the exact shape an `NqLinear` would take |
| CUDA kernels | `candle-kernels/build.rs:18-60` compiles all `.cu` via `cudaforge` into PTX, plus statically compiled MMQ/MMVQ/MoE sources `:31-45`; feature `cuda` at `candle-core/Cargo.toml:52` | new `.cu` + build.rs entry + `cuda_fwd`; hours, not days |
| Metal kernels | `candle-metal-kernels/src/kernels/quantized.rs`; feature `metal` at `Cargo.toml:58-62`; `metal_fwd` per op | new MSL source + Rust binding |
| custom-op example | `candle-examples/examples/custom-ops/main.rs` (RMSNorm, CPU+CUDA, forward+backward) | complete worked example |

**Verdict:** a CPU NQ op is cheap and needs no `candle-kernels` change. `NqLinear` owns
`Arc<NqPack>`, implements `Module` via `xs.apply_op1(self)`; the expensive part is forking
`quantized_qwen3.rs` (568 lines, `QMatMul` in 7 fields, `:52-54,156-159,455`).

### 1.3 Training

- Autograd: matmul backward is implemented (`candle-core/src/backprop.rs:459-470`), so `Linear`,
  attention and FFN train on CPU.
- Optimizers: only SGD (`candle-nn/src/optim.rs:35`) and AdamW (`:111`); no grad accumulation,
  clipping, LR schedule or mixed-precision master weights. Worked CPU training example:
  `candle-examples/examples/mnist-training/main.rs:149` (`backward_step`).
- Quantized weights are **not** differentiable: `QMatMul` forwards with `apply_op1_no_bwd`
  (`quantized/mod.rs:1313`). Any refit of an NQ base must hold the factors as ordinary `Tensor`s
  (low-rank `A@B` plus sparse residual), which is straightforward; only the fused inference op is
  non-differentiable.
- LoRA: **absent** (no `lora` module; the string appears only in unrelated model code).
- QAT: no built-in fake-quant, but `CustomOp1::bwd` can return `grad_res` unchanged
  (`custom_op.rs:39`), which is a straight-through estimator in ~30 LOC. No example.
- Memory floor, 5.4B FFN: F32 weights + grads + AdamW moments = 16 B/param = 86 GB for the whole
  FFN; one 151M-param layer = 2.4 GB. Layer-at-a-time refit fits 6.7 GB; whole-model does not.
  8B full fine-tune = 128 GB.

### 1.4 Browser

| example | what it proves | path |
|---|---|---|
| `candle-wasm-examples/quant-qwen3` | Qwen3-0.6B **GGUF** in browser, WASM SIMD128, **no WebGPU** | `src/m.rs:49` `Device::Cpu`; `.cargo/config.toml` `+simd128`; README claims 8.7 tok/s Q8_0 (645 MB), 5.8 tok/s Q4_K_M (380 MB) |
| `candle-wasm-examples/llama2-c` | 15M/42M/110M llama2.c, F32 `.bin`, fetched as a static asset | `src/app.rs:57-58` `fetch_url("model.bin")`; CPU only (`worker.rs:68`) |
| `phi`, `t5`, `whisper`, `bert`, `moondream`, `yolo`, `blip`, `segment-anything` | other in-browser demos | all `Device::Cpu` |

- Candle has **no WebGPU backend at all**: `Device` is only `Cpu | Cuda | Metal`
  (`candle-core/src/device.rs:16-20`), and no `wgpu` code appears anywhere in the repository.
- Candle's tiny-LM answer, recorded for comparison only: wasm32-unknown-unknown + SIMD128. A
  10-100M Q8 model is 10-100 MB; the 4 GB wasm32 linear-memory ceiling is the hard limit. Candle
  materialises the whole GGUF plus F32 embeddings, so 0.6B is comfortable and larger models
  approach the ceiling. apeiron does not take this path (§3.3).
- Crates: `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers` (`unstable_wasm`),
  `wasm-bindgen`, `js-sys`, `web-sys`, `getrandom 0.3` (`wasm_js`), `console_error_panic_hook`;
  `-C target-feature=+simd128`. No threads (rayon collapses to one), no mmap, model in memory.

### 1.5 Health

MIT OR Apache-2.0. 0.11.1 (`Cargo.toml:26`); `CHANGELOG.md` stale (v0.3.1, 2023). 635 Rust files,
226,644 lines. Gaps: no WebGPU/IQ/mmap, two optimizers, no LoRA/QAT.

## 2. Burn 0.23.0-pre.2

### 2.1 Backends

| backend | feature | device | maturity evidence |
|---|---|---|---|
| wgpu (native) | `wgpu` | `Device::wgpu` | default GPU path in most examples; fusion + autotune on by default (`crates/burn/Cargo.toml:141`) |
| WebGPU (browser) | `webgpu` | `Device::webgpu` (`crates/burn-tensor/src/device.rs:448-456`) | one example (MNIST); docs `burn-book/src/building-blocks/backend.md:31` |
| CUDA | `cuda` | `Device::cuda(0)` | CubeCL, examples/tests |
| ROCm | `rocm` | `Device::rocm(0)` | CubeCL HIP |
| Metal | `metal` | `Device::metal` | CubeCL native Metal; `wgpu-msl` for MSL-through-wgpu |
| CPU (JIT) | `cpu` | `Device::cpu()` | CubeCL CPU runtime, LLVM JIT, fusion/autotune (`burn-cpu/README.md:10-30`) |
| CPU (portable) | `flex` | `Device::flex()` | pure Rust, SIMD + rayon, no_std/WASM (`burn-flex/Cargo.toml:18-36`) |

`burn-ndarray` no longer exists in this tree; `burn-flex` replaced it (40 crates under `crates/`).
All backend features are in `crates/burn/Cargo.toml:134-142`.

### 2.2 Quantization

- Schemes are CubeCL's, not ggml's: `QuantValue::{Q8F,Q4F,Q2F,Q8S,Q4S,Q2S,E5M2,E4M3,E2M1}`,
  per-tensor/per-block scales in F32/F16/BF16/UE4M3, storage Native/PackedNative/PackedU32
  (`burn-std/src/tensor/quantization/base.rs:2`; table in
  `burn-book/src/performance/quantization.md:117-150`). **No Q4_K/Q5_K/Q6_K, no superblocks.**
- Kernels: quantize/dequantize at `crates/burn-cubecl/src/kernel/quantization/{quantize,dequantize}.rs`;
  Flex reference at `crates/burn-flex/src/ops/qtensor.rs:125-212`.
- Matmul: default `q_matmul` dequantizes both operands then calls float matmul
  (`burn-backend/src/backend/ops/qtensor.rs:499-541`). CubeCL overrides it to pass quantized storage
  into cubek's tile matmul (`burn-cubecl/src/ops/qtensor.rs:359-397`;
  `burn-cubecl/src/tensor/stored.rs:18-46`), so there is a packed path on wgpu/cuda, but the set of
  schemes cubek accepts is external to this repo and **not verified**. Flex has no `q_matmul`
  override, so `Device::flex()` dequantizes per call.
- GGUF import: **absent**. `burn-store` reads burnpack/SafeTensors/PyTorch
  (`burn-store/src/lib.rs:10-20`); the only "GGUF" is an alignment comment (`burn-pack/src/base.rs:42`).
  No Qwen3/Llama module exists either; assemble one from `TransformerEncoder`/`MHA`/RMSNorm/RoPE.
- **Can Burn load a Qwen3 GGUF today? No.** No parser, no K-quant layout, no Qwen3 module, no GGUF
  tokenizer path. It would need GGUF→Q4S/Q8S requantisation, which changes the quantisation grid.
- Immaturity markers: quantized reshape has `todo!`/`unimplemented!` for sub-byte and block
  recomputation (`burn-cubecl/src/ops/base.rs:470-490`).

### 2.3 Extension points

- `Backend` = `BackendTypes` + `FloatTensorOps` + `QTensorOps` + `ModuleOps` + ... 
  (`burn-backend/src/backend/base.rs:121-139`).
- `#[backend_extension]` generates dispatch for `Autodiff`, `Fusion`, `Cube`, `Flex`, `Remote`;
  implement the trait per backend, optionally with a handwritten backward
  (`crates/burn-backend-extension/README.md`; tutorials
  `burn-book/src/advanced/backend-extension/{custom-cubecl-kernel,custom-wgpu-kernel}.md`; worked
  examples `examples/custom-cubecl-kernel` and `examples/custom-wgpu-kernel`).
- Cost of an NQ op: define one extension trait over `(x, a, b, sparse_idx, sparse_val)`; a Flex
  reference can compose existing float ops, a Cube kernel is WGSL/CUDA via CubeCL. 200-400 LOC for
  CPU, plus a handwritten bwd for training. The expensive part is the data path and model, not the
  op. Float reference plus optimised kernels is the intended split (`crates/burn-backend-tests`).

### 2.4 Training

- Autograd for matmul/linear on all backends (`crates/burn-autodiff`, with checkpointing).
- Optimizers: Adam, AdamW, SGD, AdaGrad, Adafactor, Adan, LAMB, Lion, LBFGS, Muon, RMSProp,
  momentum (`crates/burn-optim/src/optim/`), plus grad accumulation, clipping and LR schedulers.
- LoRA: implemented as a `Reparameterizer`; `Module::apply_lora` (`burn-core/src/module/lora.rs:1-50`,
  `burn-core/src/module/base.rs:470`). Gradients flow through the adapter over a frozen quantized
  base because `dequantize` starts a grad-free leaf (`burn-autodiff/src/ops/qtensor.rs:20-48`).
- QAT: **not supported**. Docs: "Burn does not currently support QAT" and "No integer operations
  are currently supported" (`burn-book/src/performance/quantization.md:24,35`). `quantize` detaches
  explicitly, with a comment that QAT would need a straight-through estimator
  (`burn-autodiff/src/ops/qtensor.rs:29-33`). An STE fake-quant op is possible through
  `#[backend_extension(Autodiff, ...)]` with a handwritten identity backward, but nothing exists.
- Linear algebra: `burn-linalg` has Cholesky/solve on CPU host and CubeCL, SVD on host
  (`crates/burn-linalg/src/{cholesky,solve_host,svd_host}.rs`); candle has none. This is the
  deciding feature for activation-aware least-squares refit.
- CPU memory: same 16 B/param F32 AdamW; LoRA over a packed base is the only 8B-in-6.7-GB route.
  Full 5.4B FFN F32 = 86 GB → 24 GB GPU layer-at-a-time, 80 GB for the whole FFN.

### 2.5 Browser

- `examples/mnist-inference-web` is the only in-browser demo. It embeds `model.bpk` with
  `include_bytes!`, runs `Device::flex()` by default and optionally `Device::wgpu` (`src/state.rs:9-18`;
  features in `Cargo.toml:13-20`). It is `#![no_std]` (`src/lib.rs:1`) and calls `__wasm_call_ctors`
  once to stop CubeCL constructors re-running per export (`src/web.rs:15-27`).
- No LLM example: `examples/text-generation` is training-only (`src/model.rs:53`), no tokenizer, no
  GGUF, so WebGPU LLM inference is **not demonstrated**. `webgpu`/`Device::webgpu` are real but
  MNIST is the only proven workload. Browser limits: 4 GB wasm32 memory, WebGPU `maxBufferSize`
  typically 256 MB-2 GB, shader-compile/autotune startup, `--cfg web_sys_unstable_apis`.
- A tiny LM can ship as a static burnpack/SafeTensors asset on Flex today; on WebGPU it needs a
  hand-written Qwen3-style model plus tokenizer, which does not exist here.

### 2.6 Health

MIT OR Apache-2.0. 0.23.0-pre.2 (`Cargo.toml:24`), CITATION 0.21.0, docs say 0.22; pre-release,
breaking changes in `changelog.md`. MSRV 1.95, edition 2024 (`crates/burn/Cargo.toml:19-26`).
1,983 files, 488,260 lines, 44 MB, 40 crates, 23 examples. Gaps: no GGUF/K-quants, no QAT, no LLM
inference/tokenizer, no WebGPU LLM demo.

## 3. Comparison

### 3.1 Serving the existing NQ 8B pack on CPU

| | apeiron engine | candle | burn |
|---|---|---|---|
| reads `.nqp` | **yes** | no (not GGUF; unknown dtype bails, `quantized/mod.rs:304-323`) | no |
| reads GGUF K-quants | yes | yes | no |
| integration | none | custom `NqLinear` (`CustomOp1`) + fork of `quantized_qwen3.rs`; 500-900 LOC, no build change | new loader + Qwen3 model + extension trait + kernels; 1.5-3k LOC |
| 8B on 6.7 GB | **yes**, 2.4 GB peak measured | no, ≈8.5 GB arithmetic (F32 embedding + repack) | no; Flex dequantizes per matmul, float weights are 16 GB |
| speed path | Q32.32 fixed-point, SIMD | AVX-512-VNNI/AVX-VNNI/AMX, NEON/i8mm, packed tiles | CubeCL LLVM JIT or Flex gemm |
| risk | own code, measured PPL | QTensor non-differentiable, no mmap, memory | pre-release API, no GGUF, unverified cubek scheme coverage |

### 3.2 Activation-aware refit / QAT for the 5.4B FFN

The chosen path is the in-tree `BlockTune` (`src/gpu/nn/train/block.rs`) and `factorize`
(`src/learn/nanoquant.rs`). The table is a capability comparison, not an adoption plan.

| | candle | burn |
|---|---|---|
| matmul autograd on CPU | yes (`backprop.rs:459-470`) | yes |
| optimizers | SGD, AdamW only | 11+, grad accumulation/clipping/LR schedules |
| linear algebra for block least squares | **none** | Cholesky/solve/SVD (`burn-linalg`) |
| built-in LoRA | no | yes (`Module::apply_lora`) |
| QAT / STE | expressible via `CustomOp1::bwd` identity, no example | not supported; `quantize` detaches; STE needs a new extension op |
| ingest NQ pack | custom reader or dense f16 (14 GB) | `.nqp`→SafeTensors converter, then `burn-store`; export back |
| 5.4B FFN CPU floor | 2.4 GB per 151M layer, 86 GB whole | same (LoRA over packed base much lower) |
| GPU | CUDA, Metal (no ROCm/WebGPU) | CUDA/ROCm/Metal/Vulkan/WebGPU |

### 3.3 In-browser tiny LM for sanguine.sh: the repo's own wasm32 build

Neither Candle nor Burn is the implementation. The browser layer compiles the existing
`Cell<i64>`/Q32.32 engine to `wasm32-unknown-unknown` behind a thin `wasm-bindgen` ABI. The core
crate stays zero-dependency; the wrapper is a workspace member, like `derive` and `fold-abi`.

| | apeiron (chosen) | candle (reference) | burn (reference) |
|---|---|---|---|
| in-browser LLM proof | none yet | Qwen3-0.6B GGUF Q4_K_M/Q8_0, wasm SIMD | none (MNIST only) |
| model input | in-memory GGUF + NQ pack (`Engine::from_bytes`, `from_bytes_with_nq`) | GGUF | burnpack/SafeTensors |
| kernel | own `Cell<i64>`/Q32.32 | ggml quant kernels | CubeCL/Flex |
| WebGPU | not needed | absent | present, MNIST-proven |
| missing work | wasm wrapper, `Bpe::from_tokenizer_bytes`, sequential pool/exact path | fork the example | write model, tokenizer, export |
| fit | one implementation, same kernel as CPU | third-party runtime | third-party runtime |

What already exists:

- `Engine::from_bytes(label, Vec<u8>)` and `Engine::from_bytes_with_nq(label, Vec<u8>, NqPack)`
  load an in-memory GGUF and NQ pack with no filesystem (`src/engine/loader.rs:966`, `:1029`).
- `Engine::step_with_kv` and `step_logits_with_mode` decode a token (`src/engine/step.rs`).
- `Bpe` parses the HuggingFace `tokenizer.json` (`src/format/bpe.rs`); it needs a
  `from_tokenizer_bytes` entry beside `from_tokenizer_json`.
- The exact Q32.32 path (`KernelMode::Exact`) needs no worker pool. The Q8Act path uses the
  `std::thread` pool (`src/engine/chart/pool.rs`); a `cfg(target_arch = "wasm32")` sequential pool
  is the speed follow-up.

What is missing:

- A `wasm/` workspace member with `crate-type = ["cdylib"]`, `wasm-bindgen`, and the API
  `load(gguf_bytes, tokenizer_bytes)`, `encode(text)`, `step(token_id)`, `decode(ids)`, `reset()`.
- `Bpe::from_tokenizer_bytes`.
- A sequential pool or exact-only first build; no `libc`, no threads.
- A tiny model asset: Qwen2.5-0.5B or Qwen3.5-0.8B Q4_K_M GGUF (~0.4–0.5 GB), or a smaller NQ
  pack, served as an optional static file. The model is downloaded on click; the page works
  without it.
- Site CSP: `connect-src 'self'` and `script-src 'self' 'wasm-unsafe-eval'`; keep it same-origin.

Candle's `quant-qwen3` is evidence that a 0.6B GGUF runs in wasm SIMD, not a dependency. Burn's
`Device::webgpu` is evidence that WebGPU works for MNIST, not an LLM. The wrapper calls the same
`Cell<i64>` word bodies and Q32.32 kernels as the CPU path; the three faces of `R[x]/(x² − σ)`
(`σ < 0`, `σ = 0`, `σ > 0`) are the algebra, and the wrapper does not fork them or the kernel. No JS
matmul and no third-party tensor runtime.

## 4. Bottom line

1. **Serving the NQ 8B pack: keep apeiron.** Only it reads `.nqp`; its 2.4 GB peak at ctx 512 is
   measured on the pack (`bench/cpu/1794553293-nq8b-q4km-ppl.md`). Candle cannot parse it and needs
   ~8.5 GB for 8B Q4_K_M; Burn has neither a GGUF/K-quant reader nor a low-rank+sparse type.
2. **Refit/QAT: apeiron's own `BlockTune`/`factorize`.** Burn has Cholesky/solve/SVD plus more
   optimizers and LoRA, but no QAT and `quantize` detaches; adopting it would be a second
   implementation. Use it only as a reference for a missing dense routine, not as a host. Rent
   24 GB for layer-wise block tuning, 80 GB for the full FFN.
3. **In-browser tiny LM: the repo's own wasm32 engine.** `Engine::from_bytes` and
   `from_bytes_with_nq` already load an in-memory model; add a `wasm/` wrapper,
   `Bpe::from_tokenizer_bytes`, and a sequential pool or exact-only path. Serve a 0.5B/0.8B
   Q4_K_M GGUF or NQ pack as an optional static asset. Candle and Burn are references only; neither
   ships.

## 5. Not verified

- No build or run: compile success, wasm bundle size, tok/s, startup time and peak RSS on this
  machine.
- Burn's packed matmul: which schemes cubek accepts on wgpu/cuda, and whether `Device::webgpu`
  still builds and runs for wasm32 at this commit.
- Candle's 8B memory arithmetic is derived from code; allocator behaviour and KV cache may differ.
- Whether a Qwen3-0.6B Q4_K_M GGUF plus tokenizer is an acceptable asset budget for sanguine.sh.
- Burn LoRA over a packed base at 8B/CPU scale; the autodiff mechanism is code-confirmed only.
