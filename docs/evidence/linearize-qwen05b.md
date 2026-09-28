# Linearize Qwen2.5-0.5B: Can Post-Training Attention Transfer Preserve Greedy Invariants?

## Question

Can `Qwen/Qwen2.5-0.5B` be converted from quadratic softmax attention to $O(1)$-state attention via post-training distillation without losing its greedy decoding behavior?

Specifically:
1. Does post-training attention transfer (LoLCATs method) to Gated DeltaNet (GDN) preserve greedy decision cells within the tolerance established by weight quantization (such as the 4-bit seal at 15.55% flip rate or GPTQ-int4 at 21.94% flip rate)?
2. Does a hybrid configuration (Configuration A: converting 18 of 24 layers, retaining 6 softmax layers in a 3:1 ratio as in Qwen3-Next) rescue greedy prefix stability and long-range associative recall compared to full conversion (Configuration B: all 24 layers)?
3. What is the measured state memory footprint per token at context length $t = 4096$ across FP16 and integer Q32.32 representations?

---

## Method & Architecture

### Target Mechanism: Gated DeltaNet (GDN)
- **Mathematical formulation**:
  For head $h \in \{0, \dots, 13\}$ with head dimensions $d_k = 64, d_v = 64$:
  $$\beta_t = \sigma(\text{Linear}_{\beta}(x_t)) \in (0, 1)^{14}$$
  $$g_t = -\text{softplus}(\text{Linear}_{\alpha}(x_t)) \le 0, \quad \alpha_t = \exp(g_t) \in (0, 1]^{14}$$
  $$q_t = \frac{W_q x_t}{\|W_q x_t\|_2}, \quad k_t = \frac{W_k x_t}{\|W_k x_t\|_2}, \quad v_t = W_v x_t$$
  The state recurrence per head is:
  $$v'_t = \beta_t (v_t - S_{t-1}^T k_t)$$
  $$S_t = \alpha_t S_{t-1} + k_t (v'_t)^T \in \mathbb{R}^{d_k \times d_v}$$
  $$o_t = \frac{1}{\sqrt{d_k}} q_t S_t \in \mathbb{R}^{d_v}, \quad y_t = W_o \text{vec}(o_t)$$
- **Initialization**:
  - $W_q, W_k, W_v, W_o$ are initialized directly from the pretrained `Qwen2Attention` weights.
  - Gate $\alpha$: `alpha_proj` initialized with weight 0 and bias -5.0, yielding $\alpha_0 = \exp(-\text{softplus}(-5.0)) \approx 0.9933 \approx 1$.
  - Gate $\beta$: `beta_proj` initialized with weight 0 and bias 0, yielding $\beta_0 = \sigma(0) = 0.5$.
  - Rotary Positional Embedding (RoPE) is omitted from GDN layers, following standard Gated DeltaNet design.
- **Reference & Kernel Implementations**:
  - `HazyResearch/lolcats` (`/home/socol/Workspace/ref/lolcats`): attention transfer distillation architecture and two-stage training paradigm.
  - `fla-org/flash-linear-attention` (`/home/socol/Workspace/ref/flash-linear-attention`): Triton chunked kernel `fla.ops.gated_delta_rule.chunk_gated_delta_rule` for chunked parallel forward/backward and naive recurrent reference.

### Configurations Tested
1. **Original FP16 Reference**: Unmodified `Qwen/Qwen2.5-0.5B` baseline (all 24 layers softmax attention).
2. **Configuration A (Hybrid 18 GDN / 6 Softmax)**:
   - 18 layers converted to Gated DeltaNet.
   - 6 evenly spaced layers kept as original softmax attention: layers `[3, 7, 11, 15, 19, 23]` (3:1 ratio).
3. **Configuration B (Full 24 GDN)**:
   - All 24 transformer blocks converted to Gated DeltaNet ($O(1)$ recurrent state per token).

### Distillation Protocol
- **Dataset**:
  - Public corpus: `HuggingFaceTB/smollm-corpus` (`fineweb-edu-dedup/train-00000-of-00234.parquet`).
  - Tokenized into 58,594 sequence blocks of $SEQ=512$ tokens (30,000,128 tokens total).
- **Stage 1 (Attention Transfer / Layer-wise MSE)**:
  - Objective: $\min \sum_{l} \text{MSE}(\hat{y}_l, y_l^*)$ where $y_l^*$ is teacher softmax attention output and $\hat{y}_l$ is student GDN output.
  - Trained tokens: 10,240,000 tokens (2,500 steps, batch size 8, seq length 512).
  - Optimizer: AdamW, peak learning rate $1.0 \times 10^{-3}$, cosine decay, 100 warmup steps.
  - Wall-clock time: 228.59 s (~45,000 tok/s on NVIDIA RTX 5080).
- **Stage 2 (LoRA Next-Token Cross-Entropy)**:
  - Objective: Causal next-token language modeling cross-entropy loss $\mathcal{L}_{\text{xent}}$.
  - LoRA rank $r = 8, \alpha = 16$ applied to projection matrices $W_q, W_k, W_v, W_o$. Gate projections $\alpha_{\text{proj}}, \beta_{\text{proj}}$ jointly fine-tuned.
  - Trained tokens: 5,120,000 tokens per configuration (2,500 steps, effective batch size 4, seq length 512).
  - Optimizer: AdamW, peak learning rate $2.0 \times 10^{-4}$, cosine decay, 100 warmup steps.
  - Wall-clock time:
    - Configuration A: 244.13 s (~21,100 tok/s).
    - Configuration B: 376.32 s (~13,600 tok/s).

---

## Evaluation Suite

Reusing the protocol and held-out slice from `docs/evidence/decision-cell-census.md` (`/home/socol/Workspace/fold/hashtrinity/prose8.txt`):
- **Held-Out Corpus**: 33 blocks of 512 tokens ($33 \times 511 = 16,863$ next-token evaluation tokens).
- **Metrics**:
  - `bits/token`: Held-out cross-entropy loss divided by $\ln 2$.
  - `flip rate`: Top-1 argmax prediction mismatch rate against original FP16 reference under teacher forcing.
  - `share of flips in D1`: Percentage of flips occurring in the lowest logit-margin decile ($m_t \le 0.125$).
  - `mean identical greedy prefix`: Length of exact match prefix before divergence across 50 held-out prompts of length 32 generating 64 tokens greedily (`do_sample=False`).
  - `Needle-In-A-Haystack (NIAH)`: Exact recall of a 5-token numerical key in a 4,096-token context at insertion distances 512 and 2048 (20 deterministic trials each).

---

## 1. Decision-Cell & Greedy Invariance Summary Table

Comparison of converted models against the unmodified FP16 reference and the weight quantization baselines from `docs/evidence/decision-cell-census.md`:

| Model / Ansatz | bits/token | flip rate | share flips in lowest decile (D1) | share flips in D1 + D2 | mean identical greedy prefix |
|---|---|---|---|---|---|
| **FP16 reference** | 4.5911 | 0.00% | 0.00% | 0.00% | 64.00 |
| *4-bit seal (quant baseline)* | 4.6726 | 15.55% | 32.38% | 57.44% | 8.66 |
| *GPTQ-int4 (quant baseline)* | 4.8137 | 21.94% | 25.95% | 46.96% | 4.78 |
| *GPTQ-ternary L=3 (quant baseline)* | 4.6554 | 12.50% | 36.81% | 63.33% | 13.32 |
| *GPTQ-ternary L=2 (quant baseline)* | 5.3845 | 39.44% | 16.42% | 31.76% | 2.74 |
| *GPTQ-ternary L=1 (quant baseline)* | 22.9027 | 99.95% | 9.87% | 19.62% | 0.00 |
| **Configuration A (Hybrid 18 GDN / 6 Softmax)** | 6.2452 | 54.49% | 8.21% | 21.07% | 1.94 |
| **Configuration B (Full 24 GDN)** | 7.0567 | 65.83% | 7.16% | 18.56% | 0.46 |

---

## 2. Long-Context Needle-In-A-Haystack (NIAH) Sanity Check

Exact recall of a 5-token random key in a 4,096-token haystack context across 20 trials:

| Model | Distance 512 Recall | Distance 2048 Recall |
|---|---|---|
| **Original FP16 Reference** | 100.0% (20/20) | 100.0% (20/20) |
| **Configuration A (Hybrid 18 GDN / 6 Softmax)** | 0.0% (0/20) | 0.0% (0/20) |
| **Configuration B (Full 24 GDN)** | 0.0% (0/20) | 0.0% (0/20) |

---

## 3. Per-Token State Footprint at $t = 4096$

Calculation of memory consumption per token and total layer state at context length $t = 4096$:

| Architecture | Total State at $t=4096$ (FP16) | Bytes/Token at $t=4096$ (FP16) | Total State at $t=4096$ (Q32.32 i64) | Bytes/Token at $t=4096$ (Q32.32 i64) |
|---|---|---|---|---|
| **Original (24 Softmax Layers)** | 50,331,648 bytes (48.0 MiB) | 12,288.0 | 201,326,592 bytes (192.0 MiB) | 49,152.0 |
| **Configuration A (Hybrid 18 GDN / 6 Softmax)** | 14,647,296 bytes (13.97 MiB) | 3,576.0 | 58,589,184 bytes (55.88 MiB) | 14,304.0 |
| **Configuration B (Full 24 GDN)** | 2,752,512 bytes (2.63 MiB) | 672.0 | 11,010,048 bytes (10.50 MiB) | 2,688.0 |

*Derivations*:
- Softmax KV cache: $2 \times 2 \times 64 = 256$ elements per token per layer ($O(t)$). At $t = 4096$, 1,048,576 floats/layer.
- GDN state: $14 \times 64 \times 64 = 57,344$ elements per layer ($O(1)$ fixed state). At $t = 4096$, 57,344 floats/layer.
- Configuration B achieves a 94.5% state memory reduction (from 48.0 MiB down to 2.63 MiB in FP16), scaling as $O(1)$ as $t \to \infty$.
- Configuration A achieves a 70.9% state memory reduction (from 48.0 MiB down to 13.97 MiB in FP16).

---

## 4. Flips Binned by Reference Margin Decile

Distribution of flip counts and percentage shares across the 10 reference margin deciles:

| Model | Total Flips | D1 [0, 0.125] | D2 (0.125, 0.25] | D3 (0.25, 0.38] | D4 (0.38, 0.62] | D5 (0.62, 0.81] | D6 (0.81, 1.12] | D7 (1.12, 1.50] | D8 (1.50, 2.00] | D9 (2.00, 3.25] | D10 (> 3.25) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Config A (Hybrid)** | 9,189 | 8.21% | 12.86% | 11.03% | 17.54% | 11.36% | 9.74% | 8.96% | 7.31% | 6.93% | 6.05% |
| **Config B (Full GDN)** | 11,101 | 7.16% | 11.40% | 10.03% | 16.41% | 10.98% | 9.75% | 9.31% | 8.13% | 8.55% | 8.29% |

---

## 5. Findings & Verdict (Grade: `measured`)

1. **Greedy Invariant Preservation: Negative Result**:
   - **Neither configuration keeps greedy behavior within the 4-bit seal's flip rate.**
   - While the 4-bit seal preserves a 15.55% flip rate (and GPTQ-int4 achieves 21.94%), **Configuration A suffers a 54.49% flip rate** and **Configuration B suffers a 65.83% flip rate**.
   - Under free-running greedy generation, the mean identical prefix drops from 64.00 tokens (reference) to **1.94 tokens** for Configuration A and **0.46 tokens** for Configuration B.
2. **Decile Concentration Failure**:
   - For high-fidelity weight quantization (such as the 4-bit seal or GPTQ $L=3$), flips concentrate heavily in D1 (32.38% and 36.81% respectively).
   - In contrast, linear attention conversion shows complete decile dispersion: only 8.21% (Config A) and 7.16% (Config B) of flips occur in D1, while over 78% of flips occur in regions where the teacher model had substantial margin gaps ($m_t > 0.25$).
   - This proves that the representation shift from linear state recurrence introduces massive global distortion throughout the logit manifold, rather than localized boundary noise.
3. **Associative Recall Breakdown**:
   - In the long-context needle-in-a-haystack test (4,096 tokens), the original model achieves 100.0% exact recall at both 512 and 2048 token distances.
   - Both converted models drop to **0.0% exact recall** across all 20 trials. Post-training attention transfer fails to preserve precise key-value retrieval over multi-thousand token contexts.
4. **Architectural Implication for Apeiron**:
   - Post-hoc distillation of pretrained softmax transformers to recurrent $O(1)$ linear attention cannot substitute for exact softmax or certified sparse softmax if greedy discrete decisions must be preserved.
   - Subquadratic inference via post-training conversion requires either:
     a) True native pretraining from scratch with linear recurrence, or
     b) Certified sparse softmax attention (`docs/epics/subquadratic-decode.md`), which mathematically preserves exact softmax outputs without reading every key.

---

## 6. Verification Artifacts & Script Inventory

- **Census & Eval JSON**: `/home/socol/oc-tmp/conv-t0/eval_results.json`
- **Gate Checker**: `/home/socol/oc-tmp/conv-t0/check.py` (sha256: `a0ccf6e6a9cf698eb312f6584fb2e9c7a69c96dcdabeccb927529e7593c0206e`)
- **Evaluation Script**: `/home/socol/oc-tmp/conv-t0/evaluate.py` (sha256: `fff9cede987e37d76f0bdcda6420fea3585060917e2cd4b00a5860de5c9ea6e6`)
- **Data Preparation Script**: `/home/socol/oc-tmp/conv-t0/prepare_data.py` (sha256: `a744ffb0ca4b8c990f4285f5a75b75b8846206b24ca1ffc68943f05690d2bb04`)
- **Stage 1 Training Script**: `/home/socol/oc-tmp/conv-t0/train_stage1.py` (sha256: `344b4a9829740a3042f62f4d666bc035082723051f867045f7ac57f2d59f3394`)
- **Stage 2 Training Script**: `/home/socol/oc-tmp/conv-t0/train_stage2.py` (sha256: `4410826957e5c5ae65d9f57be5b8418e064fdc1dcc0b5eaf8dfa66f8ed25b0ea`)
- **Base Model Digest**: `Qwen/Qwen2.5-0.5B` (`060db6499f32faf8b98477b0a26969ef7d8b9987`)
- **Lake Bridge Digest**: `proof/ApeironWord.lean` (sha256: `2c41dae6ffcac7faceac06a7378c66e897326d9f0b60f7349a29a680829bde71`)
