# ExLlamaV3 GPU Baseline: Speed, Quality, and Bitrate Scaling on Qwen2.5-0.5B

## Question and Context

Task `APEIRON-EXL-T0` benchmarks ExLlamaV3 as the GPU performance and quantization bar for conversion research on `Qwen/Qwen2.5-0.5B`.

ExLlamaV3 implements the EXL3 format based on Trellis/QTIP (Quantized Trellis Initialized Permutation / LDLQ error feedback with Viterbi decoding). We evaluate:
1. **Speed**: Median decode throughput (tokens/second) on batch size 1, greedy decoding from prompt `"The capital of France is"`, across 64 and 256 generated tokens over 5 repetitions, excluding prefill latency.
2. **Quality (Decision-Cell Census)**: Teacher-forced argmax flip rate, bits/token cross-entropy loss, flip concentration in the lowest logit margin decile, and mean identical greedy prefix length over 50 held-out prompts of 64 tokens, compared directly against the `dec-t0` ansätze (fp16 reference, 4-bit seal, GPTQ-int4, and GPTQ-ternary $L=1, 2, 3$).

---

## 1. Environment, Commits, and Hardware State

### Hardware and System
- **Host CPU**: Intel Core i9-14900KF (24 cores / 32 threads, 8P + 16E)
- **GPU**: NVIDIA GeForce RTX 5080 (16,303 MiB physical VRAM, sm_120, Blackwell architecture)
- **Driver**: NVIDIA Linux x86_64 Driver `615.71.09`, CUDA UMD `13.4`
- **System CUDA Toolkit**: CUDA `13.4` at `/opt/cuda` (`nvcc: release 13.4, V13.4.92`)
- **Python / PyTorch**: Python 3.12.13 in isolated venv `/home/socol/oc-tmp/exl3/.venv`, PyTorch `2.11.0+cu128` (CUDA capability `(12, 0)`, sm_120 support verified in arch list)

### Software Commits and Artifacts
- **ExLlamaV3**: git clone `https://github.com/turboderp-org/exllamav3` commit `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`
  - Release wheel installed: `exllamav3-1.5.3+cu128.torch2.11.0-cp312-cp312-linux_x86_64.whl` (turboderp release v1.5.3 matching PyTorch 2.11.0+cu128)
- **llama.cpp**: commit `ece963f41b0b02d7a0d61436ae365762c073a4c8` (build 10450)
- **Apeiron**: checkout revision `HEAD`, binary `/home/socol/Workspace/apeiron/target/release/examples/engine_decode` (compiled with `APEIRON_KERNEL=q8act`)
- **Base Model**: `Qwen/Qwen2.5-0.5B` (`/home/socol/models/qwen2.5-0.5b`, 494,032,768 parameters total, 357,854,208 projection parameters)
- **GGUF Model**: `/home/socol/models/gguf/qwen05b-Q4_K_M.gguf` (391,859,712 bytes, Q4_K_M quantization)

### Shared GPU and Host State
The GPU is shared with display and desktop compositing workloads (~4.2 GB baseline VRAM usage). Exact memory and utilization were polled via `nvidia-smi` at every step:
- Idle before benchmarks: ~4,236 MiB used
- During EXL3 inference: ~4,985–5,139 MiB used (~750–900 MiB footprint for model weights and KV cache)
- Host 1-minute load average during CPU benchmarks: ~2.5 to 5.9 (shared host under background agent load)

---

## 2. EXL3 Model Conversion

Models were converted from `/home/socol/models/qwen2.5-0.5b` using ExLlamaV3's `convert.py` with default calibration dataset (250 rows × 2048 cols):
```sh
/home/socol/oc-tmp/exl3/.venv/bin/python /home/socol/oc-tmp/exl3/exllamav3/convert.py \
  -i /home/socol/models/qwen2.5-0.5b \
  -o /home/socol/oc-tmp/exl3/models/qwen2.5-0.5b-exl3-<bpw>bpw \
  -w /home/socol/oc-tmp/exl3/work-<bpw> \
  -b <bpw>
```

Conversion wall times:
- **4.0 bpw**: 115.67 s (1 min 55 s), safetensors size: 554,917,602 bytes
- **3.0 bpw**: 110.17 s (1 min 50 s), safetensors size: 510,189,282 bytes
- **2.0 bpw**: 104.92 s (1 min 45 s), safetensors size: 465,460,962 bytes

Storage breakdown per model:
- All 168 transformer projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) are quantized into Trellis codes with input/output scales (`suh`, `svh`).
- The output `lm_head` is quantized to 6.0 bpw by default in all configurations.
- Embeddings (`embed_tokens`) and norms (`input_layernorm`, `post_attention_layernorm`, `norm`) remain in 16-bit.
- Measured layer bits/weight across the 357,854,208 quantized projection parameters:
  - 4.0 bpw target: **4.03 b/w** (trellis: 4.00 b/w)
  - 3.0 bpw target: **3.03 b/w** (trellis: 3.00 b/w)
  - 2.0 bpw target: **2.03 b/w** (trellis: 2.00 b/w)

---

## 3. Speed Benchmarks

### Protocol
- **Prompt**: `"The capital of France is"` (5 tokens in Qwen tokenizer, 6 with BOS)
- **Batch Size**: 1
- **Decoding**: Greedy (`GreedySampler` in EXL3, argmax in llama.cpp and Apeiron)
- **Output lengths**: 64 tokens and 256 tokens
- **Reps**: 5 repetitions (median decode tokens/second reported, excluding prefill)
- **Commands**:
  - **exllamav3**: Ran via `/home/socol/oc-tmp/exl3/bench_speed.py` invoking `exllamav3.Generator` with `Job` and `GreedySampler` (timing `time_generate` over `new_tokens`).
  - **llama.cpp CUDA**: `/home/socol/Workspace/llama.cpp/build-cuda/bin/llama-bench -m /home/socol/models/gguf/qwen05b-Q4_K_M.gguf -ngl 99 -p 0 -n 64 -r 5 -o json` (and `-n 256`).
  - **llama.cpp CPU**: `/home/socol/Workspace/llama.cpp/build-cpu/bin/llama-bench -m /home/socol/models/gguf/qwen05b-Q4_K_M.gguf -p 0 -n 64 -r 5 -t 8 -o json` (and `-n 256`).
  - **Apeiron CPU**: `APEIRON_KERNEL=q8act /home/socol/Workspace/apeiron/target/release/examples/engine_decode /home/socol/models/gguf/qwen05b-Q4_K_M.gguf /home/socol/models/qwen2.5-0.5b/tokenizer.json <n>` (3 reps; median reported).

### Speed Results Table

| engine | device | format | bpw | tok/s 64 | tok/s 256 |
|---|---|---|---|---|---|
| exllamav3 | RTX 5080 (CUDA) | EXL3 4.0 bpw | 4.0 | 460.13 | 460.51 |
| exllamav3 | RTX 5080 (CUDA) | EXL3 3.0 bpw | 3.0 | 455.92 | 452.38 |
| exllamav3 | RTX 5080 (CUDA) | EXL3 2.0 bpw | 2.0 | 470.87 | 467.82 |
| llama.cpp | RTX 5080 (CUDA) | Q4_K_M | 4.5 | 872.44 | 870.74 |
| llama.cpp | Intel i9-14900KF (CPU, 8 threads) | Q4_K_M | 4.5 | 105.21 | 106.63 |
| apeiron | Intel i9-14900KF (CPU) | Q4_K_M | 4.5 | 141.57 | 127.36 |

*Device boundary note*: Speed rows on different devices (RTX 5080 GPU vs i9-14900KF CPU) represent distinct execution fabrics and are **not** a like-for-like engine comparison. They are presented together solely to anchor the hardware ceilings.

---

## 4. Quality Benchmark (Decision-Cell Census)

### Protocol
Evaluated strictly following the `APEIRON-DEC-T0` decision-cell census protocol (`docs/evidence/decision-cell-census.md`):
- **Corpus**: `/home/socol/Workspace/fold/hashtrinity/prose8.txt`
- **Teacher-forced Split**: 33 held-out blocks of $SEQ=512$ ($N = 16,863$ next-token evaluations), disjoint from calibration by >600,000 tokens.
- **Reference Margin Deciles**: 10 reference margin bins based on fp16 top-1 minus top-2 logit gap ($m_t = z_{(1)} - z_{(2)}$) cutoffs: `[0.1250, 0.2500, 0.4062, 0.5938, 0.8125, 1.0781, 1.4453, 2.0156, 3.2969]`.
- **Greedy Trajectories**: 50 non-overlapping prompts of 32 tokens sampled from held-out prose; greedy rollout of 64 new tokens per prompt.

### Combined Quality Table

| Ansatz | bits/weight | bits/token | flip rate | share of flips in the lowest margin decile | mean identical greedy prefix |
|---|---|---|---|---|---|
| fp16 reference | 16.0 | 4.5913 | 0.00% | 0.00% | 64.00 |
| 4-bit seal | 4.48 | 4.6726 | 15.55% | 32.38% | 8.66 |
| GPTQ-int4 | 4.0 | 4.8137 | 21.94% | 25.95% | 4.78 |
| GPTQ-ternary L=3 | 4.75 | 4.6554 | 12.50% | 36.81% | 13.32 |
| GPTQ-ternary L=2 | 3.17 | 5.3845 | 39.44% | 16.42% | 2.74 |
| GPTQ-ternary L=1 | 1.58 | 22.9027 | 99.95% | 9.87% | 0.00 |
| EXL3 4.0 bpw | 4.03 | 4.6259 | 9.73% | 46.22% | 11.80 |
| EXL3 3.0 bpw | 3.03 | 4.7349 | 18.66% | 28.09% | 4.10 |
| EXL3 2.0 bpw | 2.03 | 5.3352 | 38.21% | 17.41% | 2.40 |

### Detailed Decile Distribution of Flips

| Ansatz | Total Flips | D1 [0, 0.125] | D2 (0.125, 0.25] | D3 (0.25, 0.41] | D4 (0.41, 0.59] | D5 (0.59, 0.81] | D6 (0.81, 1.08] | D7 (1.08, 1.45] | D8 (1.45, 2.02] | D9 (2.02, 3.30] | D10 (> 3.30) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXL3 4.0 bpw | 1,640 | 46.22% | 26.28% | 15.73% | 6.95% | 2.80% | 1.28% | 0.55% | 0.18% | 0.00% | 0.00% |
| EXL3 3.0 bpw | 3,147 | 28.09% | 22.59% | 18.43% | 13.25% | 8.01% | 4.67% | 2.99% | 1.21% | 0.51% | 0.25% |
| EXL3 2.0 bpw | 6,444 | 17.41% | 15.46% | 14.71% | 13.61% | 11.41% | 8.64% | 7.40% | 5.80% | 3.54% | 2.02% |

---

## 5. Analysis and Findings (Grade: `measured`)

1. **EXL3 4.0 bpw SOTA Pareto Frontier**:
   - At 4.03 b/w, EXL3 achieves **4.6259 bits/token** cross-entropy loss, significantly outperforming GPTQ-int4 (4.8137 b/t at 4.0 b/w), 4-bit seal (4.6726 b/t at 4.48 b/w), and even GPTQ-ternary $L=3$ (4.6554 b/t at 4.75 b/w).
   - Its flip rate is only **9.73%** (1,640 flips out of 16,863 tokens), establishing a new state of the art on this model for 4-bit quantization.
   - Crucially, **72.50%** of all flips at 4.0 bpw fall in the lowest two deciles (D1 + D2, margin $\le 0.2500$), and 46.22% in D1 alone. Only 3 flips occur in D8 (0.18%), and zero flips occur in D9 or D10. This strongly validates that Trellis/LDLQ quantization errors at 4 bits are confined to narrow decision boundaries.

2. **Degradation at Low Bitrates (3.0 and 2.0 bpw)**:
   - EXL3 at 3.03 bpw achieves **4.7349 b/tok** and an 18.66% flip rate, handily beating GPTQ-int4 (which needed 4.0 b/w to reach 4.8137 b/t) while using 1 bit less per weight.
   - However, at 2.03 bpw, EXL3 degrades substantially: cross-entropy loss rises to **5.3352 b/tok** and the flip rate reaches **38.21%** (comparable to GPTQ-ternary $L=2$ at 39.44%).
   - At 2.0 bpw, only **17.41%** of flips reside in D1, and 67.13% of all flips occur outside the lowest two deciles (margins $> 0.2500$). This confirms the substantive finding from `decision-cell-census.md`: below ~3 bits per weight, post-training quantization noise overwhelms even large confidence margins, rendering greedy rollouts unstable (mean identical prefix drops to 2.40 tokens).

3. **Inference Latency Profile**:
   - On the RTX 5080, llama.cpp CUDA achieves **872.44 tok/s** on Q4_K_M (using cuBLAS/CUTLASS GEMV and fused dequantization).
   - ExLlamaV3 delivers **455–471 tok/s** across 2.0–4.0 bpw. The decode throughput is essentially constant across bitrates (~2.1–2.2 ms per token), showing that dispatch overhead, Python Generator orchestration, and Trellis bit-unpacking kernels dominate memory bandwidth on a 0.5B model.
   - Apeiron's Q8Act CPU engine delivers **141.57 tok/s** on 64 tokens, surpassing llama.cpp CPU (105.21 tok/s) on the shared Intel i9-14900KF host.

---

## 6. Residuals and Trust Boundary

- **Shared Host & GPU**: The GPU was shared with a desktop compositor (~4.2 GB baseline VRAM), and the host CPU ran background compilation and test tasks (load 2.5–5.9). GPU utilization and host load were recorded at each test.
- **Model Architecture Scope**: All numbers are measured strictly on `Qwen/Qwen2.5-0.5B`. Larger models (7B, 14B, 70B) have higher arithmetic intensity and different margin distributions.
- **Verification Scripts**:
  - Raw JSON metrics: `/home/socol/oc-tmp/exl3/speed_results.json`, `/home/socol/oc-tmp/exl3/quality_results.json`
  - Checker gate: `/home/socol/oc-tmp/exl3/check.py`
