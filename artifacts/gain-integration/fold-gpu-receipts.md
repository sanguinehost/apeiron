# Fold GPU receipts for Apeiron reuse

```
SCOPE:      Record measured Fold SPIR-V versus CUDA results and the source kernels they can seed.
REGISTER:   Planning/evidence; no implementation, proof, or performance claim for Apeiron.
CORPUS:     No Apeiron theorem applies. Read Fold's retained result files and GPU source listed below.
REFERENCE:  /home/socol/Workspace/fold/AGENTS.md and the exact primary artifacts below.
GATE:       Read-only review; no benchmark/build run. Raw-output search found no retained bench_gemm.*.log.
GRADE:      Retained benchmark numbers are measured within their stated scopes; this synthesis is anchor.
```

Decision: reuse Fold's Vulkan/SPIR-V GEMV and decode implementations as the GPU starting point. The existing evidence includes GEMV kernels within about 5% of cuBLAS f16, plus a Qwen 27B serving path at 88% of llama.cpp CUDA effective bandwidth and 94% of its Vulkan tok/s on the same model file. This supports a concrete baseline for the GPU epic; it does not establish a universal CUDA parity claim.

## Direct kernel comparison

`/home/socol/Workspace/fold/hashtrinity/bench/results_gpu.633ea8b.md:75-87` retains batch-1 matched-shape GEMV measurements on an RTX 5080 (Blackwell sm_120):

| Shape M×K | cuBLAS f16 | Fold SPIR-V f16 | f16 delta | Fold q4dp4a | q4dp4a vs f16 cuBLAS |
|---|---:|---:|---:|---:|---:|
| 8192×8192 | 154.1 µs | 155.9 µs | +1.2% slower | 61.0 µs | 2.53× faster |
| 4096×14336 | 131.1 µs | 137.8 µs | +5.1% slower | 53.0 µs | 2.47× faster |

The f16 rows compare matching input precision and logical GEMV shapes; the q4 rows compare the same shape while Fold reads packed 4-bit weights and cuBLAS uses f16. The latter is a useful measured carrier result, not an equal-format comparison. The retained protocol names `nvcc -O3 -arch=sm_120`, cuBLAS, boost warmup, L2-defeating weight copies and best-of-N. `hashtrinity/scripts/bench_gemm.sh` compiles `bench/cublas_gemm.cu` and the `simt_gemv` example; the example calls `simt::{gemv_bytes,gemv_dp4a_bytes}` (`/home/socol/Workspace/fold/modelfold/examples/simt_gemv.rs:1-24`). The implementation builds a Vulkan compute pipeline from SPIR-V (`/home/socol/Workspace/fold/modelfold/src/simt/gemv.rs:7-20`, `/home/socol/Workspace/fold/modelfold/src/gpu/runtime.rs:36-48`).

The benchmark summary is retained, while the generated `bench_gemm.<commit>.log` raw output is absent from the current Fold tree. Its source prints GPU name and driver, but the retained summary does not provide the exact driver/toolkit versions or the individual repetitions for this GEMV table. Do not fill those fields from the current machine.

## Model-scale decode comparison

`/home/socol/Workspace/fold/results/vs-llamacpp-27b.md:10-28` records the same Qwen3.8-27B-UD-IQ2_M GGUF on an RTX 5080, with llama.cpp build `ece963f41` (10450), CUDA and Vulkan builds, and Fold `gpu_decode_bench --gen 128 --reps 3`:

| Backend | Model bytes | tok/s | Effective GB/s |
|---|---:|---:|---:|
| llama.cpp CUDA | 10.31 GB | 67.97 ± 0.18 | 701 |
| Fold Vulkan/SPIR-V, serving recipe | 13.02 GB | 47.6 | 620 |
| llama.cpp Vulkan | 10.31 GB | 50.42 ± 0.72 | 520 |

Effective bandwidth is `tok/s × bytes`, so it compares the weight-stream rate despite different model sizes. Fold reaches 620/701 = **88.4% of CUDA's effective bandwidth** (11.6% lower when the gap is expressed relative to CUDA) and 47.6/50.42 = **94.4% of llama.cpp Vulkan's tok/s** (5.6% lower), while streaming 26% more bytes. Against the same-API Vulkan baseline Fold's effective bandwidth is 19% higher. This is a serving-chain and model-recipe measurement, not an isolated-kernel comparison. The retained gate says zero device/host argmax mismatches over 128 warm steps and the French-capital prompt (`vs-llamacpp-27b.md:14-15`).

The Qwen result's earlier 46.3 tok/s run is a separate recipe/state: `/home/socol/Workspace/fold/results/qwen38-27b-gpu.md:17-32` names `SERVEFOLD_Q2_KEEP=qkv,o,lm`, 12.80 GB uploaded, and 32 warm steps; the corrected direct comparison above records 47.6 tok/s and 13.02 GB. Keep the receipts separate. `/home/socol/Workspace/fold/results/gpu-bandwidth.md:8-24` supplies the same-card bandwidth context: RTX 5080, 8192×8192 and 4096×14336 carrier sweeps, with f32 at 891.2 and 887.2 GB/s. `/home/socol/Workspace/fold/results/hybrid-gpu-chain.md:10-18, 21-27` records the chain behind the hybrid serving route and its standalone 0.8B gate (371.5 tok/s, 426 dispatches/token, zero argmax mismatches); that is a separate model and result.

## Reusable source and evidence limits

- The batch-1 projection primitive is Fold's hand-emitted Vulkan SPIR-V GEMV in `/home/socol/Workspace/fold/modelfold/src/simt/gemv.rs`; `Carrier` selects direct f16 or packed chart/DP4A computation. The `simt_gemv` example supplies a carrier-by-carrier correctness arm against f64-dequantized weights (`simt_gemv.rs:16-19`).
- Model-scale decode source is in Fold's `bedrock` serving chain; the retained comparison's executable is `gpu_decode_bench`. Reuse should preserve the exact model, carrier recipe, prompt/token count, backend, and metric named by that receipt.
- Fold HEAD at review: `d497161f32e0d366220f2065ec36e680bd9aed86`. The cited Fold result files and kernel sources have no working-tree modifications. SHA-256: `modelfold/src/simt/gemv.rs` `ea394fcb501c2dccfb621a57222c907780384535ba3c334da5a9d68dac15609c`; `modelfold/examples/simt_gemv.rs` `302caa5608b2094b8aa346c4fdf5e2f5f234a67a49dfc1628b94d0e358112e1f`; `hashtrinity/bench/results_gpu.633ea8b.md` `8d3c6bd9afe841feec34a4d22ced179566df7b8209ecfd3013bb3e344ee99139`; `results/vs-llamacpp-27b.md` `536b6a937515e9ae69b1722dab2a7ba85beb58a6fbd01f3be020edf44f32498c`.
- The batch-1 decode receipt names vLLM 0.24.0 and SGLang 0.5.9 for its other comparison arms (`results_gpu.633ea8b.md:3-5`); these are not the 27B CUDA/Vulkan comparison. The 27B note pins llama.cpp build `ece963f41`; it does not record a Vulkan driver version, CUDA toolkit version, or raw per-run Fold output. The searched Fold tree has no `bench_gemm.*.log` or matching GPU JSON. No benchmark was rerun for this review.

## Apeiron application

Seed the GPU epic with the existing Vulkan/SPIR-V GEMV kernel family and retain these three separate checks: matched-shape f16-versus-cuBLAS, packed q4dp4a-versus-f16-cuBLAS (explicitly format-asymmetric), and model-scale Vulkan serving versus both llama.cpp backends on the identical model file. Treat the current figures as historical measured baselines at Fold's recorded source revision; the GPU epic still needs its own source/tool identity and fresh acceptance measurements before making an Apeiron performance claim.
