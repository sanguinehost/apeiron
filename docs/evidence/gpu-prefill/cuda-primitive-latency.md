# llama.cpp CUDA MMQ Q8 Prefill Projection Primitive Latency

Device **primitive compute time only**. Matched measurement of llama.cpp's Q8 prefill projection
(MMQ int8 MMA path) at Apeiron native-kernel projection shapes, directly comparable to Apeiron's
measured 171.63–506.36 GMAC/s (`docs/evidence/gpu-prefill/primitive-latency.md`).

- Command: `python3 tools/gpu/cuda_q8_gate.py` (cwd repository root)
- Result: exit 0, elapsed 3.09 s, `PASS`.
- Source identity:
  - `tools/gpu/cuda_q8_bench.cpp` sha256 `647b091e…be0150`
  - `tools/gpu/cuda_q8_gate.py`
  - llama.cpp commit `ece963f41b0b02d7a0d61436ae365762c073a4c8` (`build-cuda/bin/libggml*.so`)
- Gate: `tools/gpu/cuda_q8_gate.py` builds the C++ driver, measures all 10 shapes, and runs four controls
  (token scaling, block scaling, CPU vs GPU numerical verification with seeded defect rejection,
  and direct Q8_1 activation rejection).

## Instrument

- Device: **NVIDIA GeForce RTX 5080**, compute capability 12.0, driver VMM yes, VRAM 15899 MiB.
- Metric: `device_ns` = CUDA event elapsed time (`cudaEventElapsedTime`) recorded on the backend
  CUDA stream directly around `ggml_backend_graph_compute_async`, with device synchronization prior
  and event synchronization post; `reps = 5`, median reported.
- Exclusions: Host allocation, context setup, weight initialization/upload, and host buffer readback
  are outside the timed region.

## Formula (per dispatch)

```text
blocks   = K / 32
MACs     = M * tokens * blocks * 32  (= M * tokens * K)
GMAC/s   = MACs / device_ns
```

## MMQ Path Isolation & Activation Format

In llama.cpp's CUDA backend (`ggml/src/ggml-cuda/`):
1. For Q8_0 weights and `tokens > 8`, `ggml_cuda_should_use_mmvq` returns false (`ne11 > 8`), and
   `ggml_cuda_should_use_mmq` returns true on Turing or newer architectures (`turing_mma_available(cc)`).
   The backend routes the operation to `ggml_cuda_mul_mat_q` (`mmq.cu:85`).
2. The activation tensor passed into `ggml_mul_mat` on CUDA must be `GGML_TYPE_F32`. Passing a direct
   `GGML_TYPE_Q8_1` activation tensor on CUDA triggers an assertion failure (`GGML_ASSERT`), as verified
   by the gate's negative control.
3. Inside `ggml_cuda_mul_mat_q`, the activation tensor is quantized on-device into `block_q8_1_mmq`
   (`QK8_1_MMQ = 128`, 4 Q8_1 blocks with interleaved layout) via `quantize_mmq_q8_1_cuda` on the CUDA
   stream, followed immediately by the int8 MMA kernel (`launch_mul_mat_q<GGML_TYPE_Q8_0>`).
4. Device timing includes both the on-device `block_q8_1_mmq` quantization and the int8 MMA GEMM.

## Results (`reps = 5`, median unless noted)

| shape | M | tokens | blocks | dev_min ns | dev_med ns | dev_max ns | MACs | GMAC/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_q | 4096 | 200 | 32 | 24672 | 24864 | 25824 | 838860800 | 33737.97 |
| attn_q | 4096 | 512 | 32 | 39904 | 41120 | 41248 | 2147483648 | 52224.80 |
| attn_k_v | 512 | 200 | 32 | 15584 | 16416 | 16576 | 104857600 | 6387.52 |
| attn_k_v | 512 | 512 | 32 | 18304 | 18560 | 18720 | 268435456 | 14463.12 |
| attn_output | 1024 | 200 | 64 | 20544 | 20736 | 21920 | 419430400 | 20227.16 |
| attn_output | 1024 | 512 | 64 | 28896 | 30656 | 31072 | 1073741824 | 35025.50 |
| ffn_gate_up | 3584 | 200 | 32 | 22080 | 22496 | 23872 | 734003200 | 32628.17 |
| ffn_gate_up | 3584 | 512 | 32 | 36672 | 37056 | 40576 | 1879048192 | 50708.34 |
| ffn_down | 1024 | 200 | 112 | 24608 | 24832 | 25984 | 734003200 | 29558.76 |
| ffn_down | 1024 | 512 | 112 | 38144 | 39168 | 39264 | 1879048192 | 47974.07 |

## Matched Comparison vs Apeiron SPIR-V Primitive

| shape | M | tokens | blocks | MACs | Apeiron dev_ns | Apeiron GMAC/s | CUDA dev_ns | CUDA GMAC/s | Ratio (CUDA/Apeiron) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_q | 4096 | 200 | 32 | 838860800 | 4887744 | 171.63 | 24864 | 33737.97 | 196.57x |
| attn_q | 4096 | 512 | 32 | 2147483648 | 12168704 | 176.48 | 41120 | 52224.80 | 295.92x |
| attn_k_v | 512 | 200 | 32 | 104857600 | 503072 | 208.43 | 16416 | 6387.52 | 30.65x |
| attn_k_v | 512 | 512 | 32 | 268435456 | 1223840 | 219.34 | 18560 | 14463.12 | 65.94x |
| attn_output | 1024 | 200 | 64 | 419430400 | 1112128 | 377.14 | 20736 | 20227.16 | 53.63x |
| attn_output | 1024 | 512 | 64 | 1073741824 | 3045984 | 352.51 | 30656 | 35025.50 | 99.36x |
| ffn_gate_up | 3584 | 200 | 32 | 734003200 | 4119136 | 178.19 | 22496 | 32628.17 | 183.11x |
| ffn_gate_up | 3584 | 512 | 32 | 1879048192 | 10348640 | 181.57 | 37056 | 50708.34 | 279.28x |
| ffn_down | 1024 | 200 | 112 | 734003200 | 1449568 | 506.36 | 24832 | 29558.76 | 58.37x |
| ffn_down | 1024 | 512 | 112 | 1879048192 | 4180608 | 449.47 | 39168 | 47974.07 | 106.73x |

## Controls

1. **Token scaling:** `device_ns` strictly increases with `tokens ∈ {200, 512}` across every shape. Outcome: `PASS`.
2. **Block scaling:** At `(M=1024, tokens=200)`, `attn_output` (blocks=64, 20736 ns) vs `ffn_down` (blocks=112, 24832 ns) strictly increases with blocks. Outcome: `PASS`.
3. **Correctness baseline:** CPU reference vs CUDA MMQ output yields `nmse = 3.73e-9`, `max_diff = 3.8e-5`. Outcome: `PASS`.
4. **Seeded defect control:** Injecting a +50.0 defect into GPU output is caught (`nmse = 0.439`) and rejected. Outcome: `RED/PASS`.
5. **Shape mismatch control:** Deliberately mismatched contraction dimensions (weight K=1024 vs act K=2048) rejected. Outcome: `RED/PASS`.
6. **Direct Q8_1 activation control:** Attempting direct Q8_1 activation on CUDA fails `GGML_ASSERT`, proving the MMQ path requires F32 activation which is quantized on-device. Outcome: `RED/PASS`.

## Boundary (obligation 11)

- This measures single-dispatch device compute time for llama.cpp's CUDA MMQ int8 GEMM.
- Excludes host scheduling overhead, graph allocation, weight upload, and host memory readback.
- Compares isolated projection latency; does not represent end-to-end token generation latency (which includes KV cache attention, normalization, and sampling).
