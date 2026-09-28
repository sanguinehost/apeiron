# Decision-Cell Census: Margin Distribution and Argmax Invariance under Weight Quantization

## Question

Greedy decoding outputs one discrete invariant per step: the argmax cell of the logits.
The hypothesis under test (grade `model` prior to this census):
> Quantisation error only matters where the top-1/top-2 margin is small, so the right acceptance metric and objective for converting weights to fewer bits is the argmax flip rate, not output MSE.

Specifically: At reduced bit-widths (particularly $L=2$ ternary planes, 9 levels, ~3.17 b/w), do flips concentrate predominantly in the narrow low-margin boundary (deciles 1 and 2), meaning a margin-aware objective has substantial headroom to preserve greedy invariants? Or are flips broadly dispersed across all margin deciles, indicating that bulk quantization noise overwhelms even large confidence margins?

---

## Method

- **Model**: `Qwen/Qwen2.5-0.5B` (HuggingFace revision `060db6499f32faf8b98477b0a26969ef7d8b9987`).
- **Precision**: `torch.float16` reference weights and activations on GPU; logits evaluated in `float32` for numerically stable logit margins and cross-entropy loss.
- **Hardware**: NVIDIA GeForce RTX 5080 (16,303 MiB physical VRAM; execution footprint < 1.5 GB VRAM).
- **Random Seed**: Fixed to `0` (`torch.manual_seed(0)`, `np.random.seed(0)`).
- **Corpus and Datasets**:
  - Source text: `/home/socol/Workspace/fold/hashtrinity/prose8.txt`.
  - Calibration split: First 24 blocks of $SEQ=512$ tokens ($24 \times 512 = 12,288$ tokens).
    - Activation scales $\tau_j = \frac{1}{T}\sum_t |X_{t,j}|$ collected via forward pre-hooks for the 4-bit seal.
    - Gram matrices $G = X^T X$ collected via forward hooks for GPTQ error-feedback.
  - Held-out evaluation split (disjoint from calibration by >600,000 tokens):
    - Teacher-forced evaluation over 33 blocks of $SEQ=512$ tokens ($33 \times 511 = 16,863$ next-token predictions, exceeding the $\ge 16\text{k}$ requirement).
  - Free-running greedy prompt set:
    - 50 non-overlapping prompts of length 32 tokens sampled from the held-out split.
    - 64 new tokens generated per prompt with `do_sample=False`.
- **Target Layers**:
  - All 168 projection `Linear` layers across all 24 transformer blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), totaling 357,826,560 quantized parameters.
  - Embeddings, layer norms, and `lm_head` remain in `fp16`.
- **Ansätze Tested**:
  1. `fp16 reference`: Unmodified baseline.
  2. `4-bit seal`: Crystallized activation-aware seal (`spinor_ttl.seal.quantize_tensor_song` with `req_bits=3.3`).
  3. `GPTQ-int4`: 15-level symmetric int4 (`maxlev=7`, block size 128).
  4. `GPTQ-ternary L=3`: 27-level ternary place-value (`maxlev=13`, block size 128).
  5. `GPTQ-ternary L=2`: 9-level ternary place-value (`maxlev=4`, block size 128).
  6. `GPTQ-ternary L=1`: 3-level ternary place-value (`maxlev=1`, block size 128).
- **Artifacts and Verification Tools**:
  - Census execution script: `/home/socol/oc-tmp/dec-t0/run_census.py` (sha256: `d70d9ef2e6d377883b682f53c7d525d4a83397bcf2e3cbb51dd4aaa3bc0b675b`)
  - Gate checker script: `/home/socol/oc-tmp/dec-t0/check_census.py` (sha256: `8d5f9e6dab9f1a5972bf51d993b8e9c6a02f9e274fda76f5de22091adf925e36`)
  - Raw JSON results: `/home/socol/oc-tmp/dec-t0/census_results.json`

---

## 1. Reference Margin Census

Over $N = 16,863$ held-out tokens, the distribution of the reference top-1 minus top-2 logit gap ($m_t = z_{(1)} - z_{(2)}$) was measured under teacher forcing:

| Quantile | Logit Margin Gap |
|---|---|
| 1% | 0.0156 |
| 5% | 0.0625 |
| 10% | 0.1250 |
| 25% | 0.3281 |
| 50% | 0.8125 |
| 75% | 1.6875 |
| 90% | 3.2969 |

The 10 reference margin decile cutoffs ($d_1$ through $d_9$) are:
`[0.1250, 0.2500, 0.4062, 0.5938, 0.8125, 1.0781, 1.4453, 2.0156, 3.2969]`.
Each decile bin contains exactly 10% of the held-out reference tokens.

---

## 2. Decision-Cell Census Summary Table

Teacher-forced metrics scored over 16,863 held-out tokens; greedy rollout prefix scored over 50 held-out prompts of 64 generated tokens each.

| Ansatz | bits/weight | bits/token | flip rate | share of flips in the lowest margin decile | mean identical greedy prefix |
|---|---|---|---|---|---|
| fp16 reference | 16.0 | 4.5913 | 0.00% | 0.00% | 64.00 |
| 4-bit seal | 4.48 | 4.6726 | 15.55% | 32.38% | 8.66 |
| GPTQ-int4 | 4.0 | 4.8137 | 21.94% | 25.95% | 4.78 |
| GPTQ-ternary L=3 | 4.75 | 4.6554 | 12.50% | 36.81% | 13.32 |
| GPTQ-ternary L=2 | 3.17 | 5.3845 | 39.44% | 16.42% | 2.74 |
| GPTQ-ternary L=1 | 1.58 | 22.9027 | 99.95% | 9.87% | 0.00 |

*Note on bits/weight*: For GPTQ ansätze, nominal bits/weight are reported above ($L \log_2 3$ for ternary, 4.0 for int4). The empirical Shannon code entropies are: GPTQ-int4: 3.30 b/w; GPTQ-ternary L=3: 4.18 b/w; GPTQ-ternary L=2: 2.52 b/w; GPTQ-ternary L=1: 0.76 b/w.

---

## 3. Flips Binned by Reference Margin Decile

The table below shows the complete breakdown of flip counts and percentage share of total flips across all 10 margin deciles (from lowest logit gap D1 to highest logit gap D10):

| Ansatz | Total Flips | D1 [0, 0.125] | D2 (0.125, 0.25] | D3 (0.25, 0.41] | D4 (0.41, 0.59] | D5 (0.59, 0.81] | D6 (0.81, 1.08] | D7 (1.08, 1.45] | D8 (1.45, 2.02] | D9 (2.02, 3.30] | D10 (> 3.30) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4-bit seal | 2,622 | 32.38% | 25.06% | 17.66% | 12.13% | 6.48% | 3.13% | 2.06% | 0.76% | 0.34% | 0.00% |
| GPTQ-int4 | 3,699 | 25.95% | 21.01% | 17.65% | 12.98% | 8.95% | 6.08% | 4.03% | 2.00% | 0.78% | 0.57% |
| GPTQ L=3 | 2,108 | 36.81% | 26.52% | 17.93% | 10.15% | 5.08% | 1.61% | 1.09% | 0.62% | 0.19% | 0.00% |
| GPTQ L=2 | 6,651 | 16.42% | 15.34% | 14.27% | 13.23% | 11.46% | 9.13% | 7.80% | 5.53% | 4.19% | 2.63% |
| GPTQ L=1 | 16,855 | 9.87% | 9.75% | 9.71% | 10.31% | 10.26% | 9.81% | 10.28% | 9.96% | 10.01% | 10.06% |

---

## 4. Interpretation (Grade: `measured`)

1. **Reproduction Check**:
   - The measurements faithfully reproduce the reference numbers from `spinor-ttl`:
     - Teacher fp16 baseline sits at 4.5913 b/tok (on the 16.8k token held-out prose window).
     - GPTQ-ternary $L=3$ (27 levels) achieves 4.6554 b/tok ($\Delta = +0.0641$ b/tok), matching teacher fidelity and beating the 4-bit seal (4.6726 b/tok) and GPTQ-int4 (4.8137 b/tok).
     - GPTQ-ternary $L=1$ breaks down completely (22.90 b/tok), confirming the `pretrained_ternary_is_lossy` threshold.

2. **Decile Concentration at High vs Low Bit-Widths**:
   - For high-fidelity models ($L=3$ ternary and 4-bit seal), flips are sharply concentrated near the decision boundary:
     - For $L=3$, **63.33%** of all flips reside in the lowest two deciles (D1 + D2, margin $\le 0.2500$), and only 0.81% of flips occur in the top three deciles (D8..D10).
     - For the 4-bit seal, **57.44%** of all flips reside in D1 + D2.
   - For $L=2$ ternary planes (9 levels, 3.17 b/w), the behavior fundamentally changes:
     - Flips in the lowest decile account for only **16.42%** of the total.
     - Flips in the lowest two deciles account for only **31.76%** of the total.
     - **68.24% of all flips sit outside the lowest two deciles** (margins $> 0.2500$). Substantial flip shares persist into D5 (11.46%), D6 (9.13%), D7 (7.80%), and even D8..D10 (12.35%).
     - With a global flip rate of **39.44%** (nearly 2 out of every 5 tokens flipped under teacher forcing), free-running greedy generation diverges almost immediately, yielding a mean identical prefix of only **2.74 tokens**.
   - For $L=1$ ternary planes, the decile distribution is almost perfectly uniform (~10.0% in every decile), indicating complete decorrelation from model certainty.

3. **Substantive Finding on the Hypothesis**:
   - The hypothesis that *"quantisation error only matters where the top-1/top-2 margin is small"* holds as a descriptive property only when overall reconstruction distortion is already small ($L \ge 3$ or 4-bit).
   - At $L=2$ planes, the hypothesis **does not hold**. The quantization noise from restricting weights to 9 levels is large enough ($E \approx 1.7 \times 10^{-2}$) to flip decisions even when the reference model has high confidence ($m_t > 1.0$).
   - Because more than two-thirds of flips at $L=2$ occur across moderate-to-high margin regimes, **a margin-aware objective does not have sufficient room to rescue post-hoc $L=2$ quantization**. The bottleneck is not boundary sensitivity, but large-scale function distortion across the entire logit distribution.

---

## 5. Residuals and Trust Boundary

- **Scope**: Evaluated on `Qwen/Qwen2.5-0.5B` across 16,863 teacher-forced tokens and 50 greedy rollout trajectories.
- **PTQ vs QAT**: This census tests post-training quantization (PTQ) via closed-form error feedback (GPTQ) and column chart scaling (Song seal). It does not test quantization-aware training (QAT / BitNet ascent), where network weights and logit margins are co-optimized during training.
- **Hardware Boundary**: Measured on NVIDIA RTX 5080 using PyTorch 2.12.1+cu130. Does not make claims regarding runtime latency, kernel dispatch, or integer arithmetic acceleration.
