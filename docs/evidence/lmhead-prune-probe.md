# LM Head Exact Certified Pruning Probe: Score-Gap Condition and Pruning Bounds

```text
SCOPE:      docs/evidence/lmhead-prune-probe.md
REGISTER:   measurement; no repository source edits; scripts and data in /home/socol/oc-tmp/lmhead-t0/
CORPUS:     proof/corpus/Nous/Seq/CertifiedSoftmax.lean (ball_bound_sound),
            proof/corpus/Numeric/PrunedArgmax.lean (pruned_argmax_exact)
REFERENCE:  docs/evidence/sparse-softmax-probe.md; docs/evidence/decode-attribution.md (L7)
GATE:       /home/socol/oc-tmp/exl3/.venv/bin/python /home/socol/oc-tmp/lmhead-t0/run_probe.py (0 exit, 832/832 exact argmax verified)
GRADE:      measured (832 real decode positions on Qwen3.5-0.8B tied LM head)
```

## 1. Intent and Hypothesis

For Qwen3.5-0.8B, the tied language model head (`token_embd.weight`, shape $248{,}320 \times 1024$, stored in `Q8_0` format at 1088 bytes per row) represents $270.2\text{ MB}$ of weight traffic per decoded token. In the bandwidth-bound CPU regime (measured at $\approx 82\text{ GB/s}$ host memory bandwidth), reading the LM head accounts for $\approx 3.30\text{ ms}$ out of the total $7.80\text{ ms}$ decode step ($128.2\text{ tok/s}$ baseline, $\approx 42\%$ of single-token decode latency and $\approx 66.6\%$ of all model weight bytes).

Unlike attention keys—which are generated dynamically per sequence and failed the rate test (`AGENTS.md` §10 row *"Addressing proposals outran their rate"*)—the tied LM head weights are **fixed static parameters**. Fixed weights pass the rate test: offline clustering, centroids, and principal component bases can be computed once ahead of time at zero per-token cost.

The governing question is therefore the **score-gap condition**:
> Can an exact certified upper bound (such that row $i$ is evaluated only if $\text{Bound}(i) \ge \max_{j} \text{logit}_j$) skip a significant fraction of vocabulary rows, or does the Cauchy–Schwarz / Hölder bound slack in $d = 1024$ dimensions force evaluating virtually the entire vocabulary?

In `docs/evidence/decode-attribution.md` (L7), a global Hölder bound on Qwen2.5 pruned only $\approx 5\%$ of rows, and $0\%$ on Bonsai ("bound $\approx 63$ logits vs median top-2 gap $0.72$, nothing pruned"). This probe measures whether localized clustering (K-means outer ball bounds) or low-rank subspace projection (PCA two-level bounds) can overcome this gap on Qwen3.5-0.8B.

---

## 2. Experimental Setup and Dataset

- **Model**: `Qwen3.5-0.8B-Q8_0.gguf` (`/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf`).
  - Vocabulary size $V = 248{,}320$, hidden dimension $D = 1024$.
  - Tied head: `token_embd.weight` ($248{,}320 \times 1024$), dequantized once to `float32` ($971.0\text{ MB}$ in RAM).
  - Normalization: `output_norm.weight` ($1024$).
- **Hidden States Data**: Real final hidden states $h = \text{rms\_norm}(x, \text{output\_norm}) \in \mathbb{R}^{1024}$ collected at each position using the bit-level reference runner (`tools/qwen35/reference.py`).
- **Positions Evaluated**: $832$ decode positions across three distinct domains:
  1. `readme_prompt`: $200$ positions from Apeiron's `README.md` prompt.
  2. `readme_gen`: $32$ greedy generated positions continuing from `README.md`.
  3. `code`: $300$ positions from Rust engine source (`src/engine/attn.rs`).
  4. `prose`: $300$ positions from plain English literature (*Pride and Prejudice* opening).
  Total positions: $200 + 32 + 300 + 300 = 832 \ge 800$.
- **Hardware & Resource Envelope**: NVIDIA GeForce RTX 5080 (Blackwell `sm_120`). Peak VRAM footprint strictly constrained to $1.02\text{ GB}$ (well below the $2.5\text{ GB}$ multi-tenant limit).
- **Artifacts under `/home/socol/oc-tmp/lmhead-t0/`**:
  - State collector: `collect_states.py`
  - Probe runner: `run_probe.py`
  - Mutation control: `control.py`
  - Raw weights: `tok_embd_f32.npy` ($971\text{ MB}$)
  - Hidden states: `h_states.npy` ($832 \times 1024$, float32)
  - Raw JSON results: `results.json`

---

## 3. Mathematical Pruning Formulations

All evaluated pruning methods are **exact**: an index or cluster is skipped if and only if its upper bound is strictly less than the true maximum logit. Greedy argmax identity is preserved by construction, in formal refinement of `proof/corpus/Nous/Seq/CertifiedSoftmax.lean` (`ball_bound_sound`) and `proof/corpus/Numeric/PrunedArgmax.lean` (`pruned_argmax_exact`).

### 3.1 Method 1: Ball Bound (Cauchy–Schwarz Outer Sphere)
The $V$ rows $\{w_i\}_{i=1}^V \subset \mathbb{R}^D$ are partitioned into $K \in \{1024, 4096, 16384\}$ clusters $C_k$ using mini-batch K-means. For each cluster $k$:
$$c_k = \frac{1}{|C_k|} \sum_{i \in C_k} w_i, \quad r_k = \max_{i \in C_k} \|w_i - c_k\|_2$$
For query $h \in \mathbb{R}^D$, Cauchy–Schwarz gives:
$$\forall i \in C_k, \quad h \cdot w_i = h \cdot c_k + h \cdot (w_i - c_k) \le h \cdot c_k + \|h\|_2 r_k \equiv \text{Bound}(k)$$
Clusters are visited in descending order of $\text{Bound}(k)$. Rows in visited clusters are evaluated, updating running maximum $m$. When the next cluster has $\text{Bound}(k) < m$, the search stops.
- Centroid scoring cost: $K \times D$ MACs.
- Row evaluation cost: $\sum_{k \in \text{visited}} |C_k| \times D$ MACs.

### 3.2 Method 2: PCA-Rotated Two-Level Bound
Rows of $W$ are projected onto the top-$d$ principal subspace ($d \in \{32, 64\}$) spanned by orthonormal columns $P \in \mathbb{R}^{D \times d}$ ($P^T P = I_d$):
$$w_{i, \text{top}} = w_i P, \quad w_{i, \text{res}} = w_i - w_{i, \text{top}} P^T, \quad R_i = \|w_{i, \text{res}}\|_2 = \sqrt{\max(0, \|w_i\|_2^2 - \|w_{i, \text{top}}\|_2^2)}$$
For query $h$:
$$h_{\text{top}} = h P, \quad \|h_{\text{res}}\|_2 = \sqrt{\max(0, \|h\|_2^2 - \|h_{\text{top}}\|_2^2)}$$
By orthogonality and Cauchy–Schwarz:
$$h \cdot w_i = h_{\text{top}} \cdot w_{i, \text{top}} + h_{\text{res}} \cdot w_{i, \text{res}} \le h_{\text{top}} \cdot w_{i, \text{top}} + \|h_{\text{res}}\|_2 R_i \equiv \text{Bound}(i)$$
Rows are filtered against a candidate maximum $m$. Rows with $\text{Bound}(i) < m$ are skipped.

---

## 4. Empirical Measurements

### 4.1 Method 1: Ball Bound (K-Means Clustering)
Results across all $832$ positions. Greedy argmax verified bit-identical ($832/832$, $100\%$ exact match):

| $K$ | Mean Radius $r$ | Max Radius $r$ | Rows Eval (Med) | Rows Eval (p90) | Rows Eval (Max) | Centroid MACs / $V$ | Total Equiv Fraction (Med) | Total Equiv (p90) |
|---|---|---|---|---|---|---|---|---|
| **1024** | $0.3577$ | $0.9215$ | $99.85\%$ | $99.85\%$ | $99.85\%$ | $+0.41\%$ | **$100.26\%$** | $100.26\%$ |
| **4096** | $0.2665$ | $0.8635$ | $99.23\%$ | $99.23\%$ | $99.23\%$ | $+1.65\%$ | **$100.88\%$** | $100.88\%$ |
| **16384** | $0.1939$ | $0.8373$ | $96.28\%$ | $96.29\%$ | $96.29\%$ | $+6.60\%$ | **$102.88\%$** | $102.89\%$ |

*Finding*: Increasing $K$ from $1024$ to $16384$ compresses cluster radius $r$ from $0.358$ to $0.194$, which allows pruning $3.72\%$ of rows at $K=16384$. However, evaluating $16{,}384$ centroids costs $6.60\%$ of full vocabulary MACs, making total arithmetic cost ($102.88\%$) **worse than dense evaluation**.

---

### 4.2 Method 2: PCA-Rotated Two-Level Bound ($d \in \{32, 64\}$)
Results across all $832$ positions. Argmax verified bit-identical ($832/832$, $100\%$ exact match):

| Dimension $d$ | Variance Explained | Mean $\|w_{\text{res}}\|_2$ | Mean $\|h_{\text{res}}\|_2$ | Mean Bound Slack | Full Dots Needed (Med) | Full Dots Needed (p90) | Seeded $m_0$ Full Dots (Med) | Seeded $m_0$ Full Dots (p90) |
|---|---|---|---|---|---|---|---|---|
| **$d = 32$** | $27.38\%$ | $0.5331$ | $126.11$ | $67.23\text{ logits}$ | **$99.69\%$** | $99.75\%$ | $99.82\%$ | $100.0\%$ |
| **$d = 64$** | $31.60\%$ | $0.5173$ | $120.93$ | $62.55\text{ logits}$ | **$99.65\%$** | $99.72\%$ | $99.80\%$ | $100.0\%$ |

*Finding*: The top-64 principal subspace explains only $31.6\%$ of the vocabulary variance. The residual norm product $\|h_{\text{res}}\|_2 \|w_{\text{res}}\|_2$ introduces an average bound slack of **$62.55$ logits**. Because the entire logit distribution has a span of only $\approx 20$ logits, the upper bound exceeds the true maximum logit for $> 99.6\%$ of the vocabulary.

---

### 4.3 Method 3: Score-Gap Statistics and Logit Distribution
Distribution of top-1 minus top-2 logit gap $\Delta = \text{logit}_{(1)} - \text{logit}_{(2)}$, and the fraction of vocabulary rows falling within $\delta \in \{5, 10, 20\}$ logits of the maximum logit $\text{logit}_{(1)}$:

#### Gap Statistics per Domain:
| Domain | Positions | Mean Gap | Median Gap | Min Gap | p10 Gap | p90 Gap | Max Gap |
|---|---|---|---|---|---|---|---|
| `readme_prompt` | $200$ | $1.16$ | $0.76$ | $0.0013$ | $0.08$ | $2.28$ | $8.58$ |
| `readme_gen` | $32$ | $3.14$ | $2.00$ | $0.2260$ | $0.58$ | $7.68$ | $9.26$ |
| `code` | $300$ | $2.68$ | $1.62$ | $0.0008$ | $0.20$ | $7.07$ | $12.80$ |
| `prose` | $300$ | $1.40$ | $0.95$ | $0.0046$ | $0.10$ | $2.99$ | $9.88$ |
| **Overall** | **832** | **1.87** | **1.08** | **0.0008** | **0.12** | **5.16** | **12.80** |

#### Fraction of Vocabulary within $\delta$ Logits of Maximum:
| Domain | $\le 5\text{ logits}$ (Med) | $\le 5\text{ logits}$ (p90) | $\le 10\text{ logits}$ (Med) | $\le 10\text{ logits}$ (p90) | $\le 20\text{ logits}$ (Med) | $\le 20\text{ logits}$ (p90) |
|---|---|---|---|---|---|---|
| `readme_prompt` | $0.015\%$ ($36$ rows) | $0.103\%$ ($255$ rows) | $0.602\%$ ($1495$ rows) | $2.857\%$ ($7095$ rows) | $60.62\%$ | $96.58\%$ |
| `readme_gen` | $0.003\%$ ($7$ rows) | $0.027\%$ ($67$ rows) | $0.094\%$ ($234$ rows) | $0.919\%$ ($2282$ rows) | $21.63\%$ | $84.05\%$ |
| `code` | $0.003\%$ ($7$ rows) | $0.035\%$ ($86$ rows) | $0.075\%$ ($185$ rows) | $0.946\%$ ($2349$ rows) | $18.25\%$ | $74.86\%$ |
| `prose` | $0.009\%$ ($23$ rows) | $0.033\%$ ($81$ rows) | $0.182\%$ ($453$ rows) | $0.774\%$ ($1922$ rows) | $26.70\%$ | $66.73\%$ |
| **Overall** | **0.008%** ($21$ rows) | **0.046%** ($114$ rows) | **0.207%** ($514$ rows) | **1.234%** ($3065$ rows) | **29.91%** | **84.85%** |

#### Overall Top-1 to Top-2 Gap Histogram ($N = 832$):
| Gap Range (logits) | Count | Fraction | Cumulative Fraction |
|---|---|---|---|
| $[0.0, 0.2)$ | $120$ | $14.42\%$ | $14.42\%$ |
| $[0.2, 0.5)$ | $118$ | $14.18\%$ | $28.61\%$ |
| $[0.5, 1.0)$ | $153$ | $18.39\%$ | $47.00\%$ |
| $[1.0, 2.0)$ | $201$ | $24.16\%$ | $71.15\%$ |
| $[2.0, 3.0)$ | $91$ | $10.94\%$ | $82.09\%$ |
| $[3.0, 5.0)$ | $59$ | $7.09\%$ | $89.18\%$ |
| $[5.0, 7.5)$ | $54$ | $6.49\%$ | $95.67\%$ |
| $[7.5, 10.0)$ | $29$ | $3.49\%$ | $99.16\%$ |
| $\ge 10.0$ | $7$ | $0.84\%$ | $100.00\%$ |

---

### 4.4 Method 4: Memory Bandwidth and Tok/s Projection (Model-Grade)

In Apeiron's measured bandwidth-bound regime on host CPU ($82\text{ GB/s}$ DDR5 memory bandwidth, baseline $7.80\text{ ms/tok} \implies 128.2\text{ tok/s}$):
- Full tied LM head read traffic: $248{,}320 \times 1088\text{ bytes} = 270.17\text{ MB}$.
- Dense LM head read latency: $270.17\text{ MB} / 82\text{ GB/s} = 3.30\text{ ms}$.
- Remainder of model (24 layers): $405.57 - 270.17 = 135.40\text{ MB} \implies 1.65\text{ ms}$.

Projected traffic and throughput taking into account metadata and centroid overheads:

| Configuration | Rows Evaluated (Med) | Head Weights Read | Centroid / Subspace Overhead | Total Head Traffic | Projected Latency | Projected Tok/s | Speedup Factor |
|---|---|---|---|---|---|---|---|
| **Dense Baseline** | $100.0\%$ | $270.2\text{ MB}$ | $0.0\text{ MB}$ | $270.2\text{ MB}$ | $7.80\text{ ms}$ | **$128.2\text{ tok/s}$** | $1.00\times$ |
| **Ball $K=1024$** | $99.85\%$ | $269.8\text{ MB}$ | $1.1\text{ MB}$ | $270.9\text{ MB}$ | $7.81\text{ ms}$ | **$128.2\text{ tok/s}$** | $1.00\times$ |
| **Ball $K=4096$** | $99.23\%$ | $268.1\text{ MB}$ | $4.5\text{ MB}$ | $272.6\text{ MB}$ | $7.83\text{ ms}$ | **$127.8\text{ tok/s}$** | $0.997\times$ |
| **Ball $K=16384$** | $96.28\%$ | $260.1\text{ MB}$ | $17.8\text{ MB}$ | $278.0\text{ MB}$ | $7.89\text{ ms}$ | **$126.7\text{ tok/s}$** | $0.989\times$ |
| **PCA $d=32$** | $99.69\%$ | $269.3\text{ MB}$ | $15.9\text{ MB}$ | $285.2\text{ MB}$ | $7.98\text{ ms}$ | **$125.3\text{ tok/s}$** | $0.978\times$ |
| **PCA $d=64$** | $99.65\%$ | $269.2\text{ MB}$ | $31.8\text{ MB}$ | $301.0\text{ MB}$ | $8.17\text{ ms}$ | **$122.4\text{ tok/s}$** | $0.955\times$ |

*Grade*: `model` (projection based on measured 82 GB/s memory bandwidth floor and measured 128.2 tok/s baseline).

---

## 5. Instrument Validation and Controls

1. **Exact Argmax Invariant (Positive Control)**:
   For all $832$ positions across all 5 configurations ($K \in \{1024, 4096, 16384\}$, $d \in \{32, 64\}$):
   $$\arg\max_{i \in \text{visited}} (h \cdot w_i) \equiv \arg\max_{i \in [0, V)} (h \cdot w_i)$$
   Total tests: $5 \times 832 = 4160$ checks; $0$ mismatches ($100\%$ bit-exact match).
2. **Seeded Defect (Negative Control)**:
   The mutation control (`/home/socol/oc-tmp/lmhead-t0/control.py`) mutates the residual bound by removing the residual slack term (subspace-only approximation, setting slack factor to $0.0$).
   - Sound bound ($1.0 \times \text{slack}$): $0 / 832$ violations.
   - Mutated bound ($0.0 \times \text{slack}$): **$832 / 832$ violations** detected (true argmax pruned on $100\%$ of positions).
   - Defect is attributable and falsifiable; instrument is validated.

---

## 6. Structural Diagnosis: Why the Score-Gap Condition Fails

The score-gap condition requires:
$$\text{Bound Slack} \le \text{True Logit Margin}$$
For pruning to skip $\ge 90\%$ of rows, the bound slack must be tighter than the 90th percentile logit spread:
- From Section 4.3, $90\%$ of vocabulary rows have logits within $\approx 15$ logits of the maximum logit.
- To prune $99\%$ of rows, bound slack must be $< 5$ logits.

However, in $D = 1024$ dimensions:
1. **Norm Inflation**: $\|h\|_2 \approx 131.8$ because the hidden states accumulate across 24 residual streams with RMS normalization scale $\approx 4.1$.
2. **Curse of Dimensionality in Ball Bounds**: Even with $K = 16{,}384$ clusters, the maximum cluster radius remains $r \approx 0.194$ (compared to average row norm $0.627$). The Cauchy–Schwarz outer product $\|h\|_2 \cdot r \approx 131.8 \times 0.194 = 25.6\text{ logits}$. Since $25.6\text{ logits}$ exceeds the entire range from min logit to max logit ($\approx 19.4\text{ logits}$), almost all cluster bounds exceed the maximum logit.
3. **Variance Dispersion in PCA**: The top-64 principal components capture only $31.6\%$ of the energy of the 248,320 embedding rows. The residual norm $\|w_{\text{res}}\|_2 \approx 0.517$ and $\|h_{\text{res}}\|_2 \approx 120.9$, yielding an average slack of **$62.55$ logits**.

Because the certified bound slack ($25.6\text{–}67.2\text{ logits}$) is an order of magnitude larger than the score gap (median $1.08\text{ logits}$) and twice the total logit spread, exact bounds cannot separate the top candidate from the tail.

---

## 7. Decision Line

**The Qwen3.5-0.8B tied LM head FAILS the score-gap condition across all tested clusterings ($K \in \{1024, 4096, 16384\}$) and principal subspaces ($d \in \{32, 64\}$).**

Exact certified pruning skips at most $0.15\%\text{–}3.72\%$ of rows, while introducing centroid and subspace metadata overheads that exceed the savings. In all cases, certified pruning yields **net negative throughput** ($0.955\times\text{–}0.999\times$ of dense baseline).

**Recommendation**: Do not implement certified bound pruning for the LM head. Dense streaming GEMV (or block-quantized Q8Act GEMV) is mathematically and architecturally optimal for this operator.

---

## 8. Bit-Plane Bound (LMHEAD-T1 Probe)

```text
SCOPE:      docs/evidence/lmhead-prune-probe.md
REGISTER:   measurement; no repository source edits; scripts and data in /home/socol/oc-tmp/lmhead-t1/
GATE:       /home/socol/opt/arena-venv/bin/python /home/socol/oc-tmp/lmhead-t1/run_bitplane_probe.py (0 exit, 832/832 exact argmax verified on Qwen3.5, 800/800 on Qwen2.5)
GRADE:      measured (832 real decode positions on Qwen3.5-0.8B Q8_0, 800 positions on Qwen2.5-0.5B Q8_0)
```

### 8.1 Intent and Mathematical Formulation

LMHEAD-T0 established that continuous geometric bounds (K-means Cauchy–Schwarz outer spheres and PCA orthogonal residual projections) fail the score-gap condition in $D=1024$ dimensions because Cauchy–Schwarz norm inflation ($\|h\|_2 \approx 131.8$) introduces $25.6\text{–}67.2\text{ logits}$ of bound slack—far wider than the total logit spread across the vocabulary ($\approx 20\text{ logits}$).

LMHEAD-T1 tests an exact discrete bound constructed directly from the format arithmetic of `Q8_0` weights without geometric relaxation.

In `Q8_0`, each 32-weight block $b$ contains a float16 scale $d_b$ (2 bytes) and 32 int8 quantized codes $q_{b,i} \in [-128, 127]$ (32 bytes), representing weights $w_{b,i} = d_b \cdot q_{b,i}$. The row size is:
$$\text{Row Size} = B \times 34\text{ bytes} \quad (B = 32 \implies 1088\text{ B/row on Qwen3.5}; \quad B = 28 \implies 952\text{ B/row on Qwen2.5})$$

#### Two-Stage Bit-Plane Decomposition
Each 8-bit signed code $q \in [-128, 127]$ is decomposed into an arithmetic upper nibble $q_{\text{hi}} \in [-8, 7]$ and an unsigned lower nibble $q_{\text{lo}} \in [0, 15]$:
$$q_{\text{hi}} = \lfloor q / 16 \rfloor \in [-8, 7] \quad (\text{arithmetic right shift by } 4), \qquad q_{\text{lo}} = q - 16 q_{\text{hi}} = q \ \& \ 0x0\text{F} \in [0, 15]$$
The dequantized dot product with query $h \in \mathbb{R}^D$ is identically:
$$w \cdot h = \sum_{b=0}^{B-1} d_b \sum_{i=0}^{31} q_{b,i} h_{b,i} = \underbrace{\sum_{b=0}^{B-1} d_b \sum_{i=0}^{31} 16 q_{\text{hi}, b, i} h_{b,i}}_{s_{\text{hi}}(\text{row})} + \underbrace{\sum_{b=0}^{B-1} d_b \sum_{i=0}^{31} q_{\text{lo}, b, i} h_{b,i}}_{\Delta(\text{row})}$$

#### Exact Hypercube Extremal Interval
Because $q_{\text{lo}, b, i}$ varies independently over $[0, 15]^{32}$, the unobserved term $\Delta(\text{row})$ is strictly bounded by element-wise sign alignment:
For each element, the contribution is $c_{b,i} q_{\text{lo}, b, i}$ where $c_{b,i} = d_b h_{b,i}$. Since $0 \le q_{\text{lo}, b, i} \le 15$:
$$\min_{q_{\text{lo}} \in [0, 15]} (c_{b,i} q_{\text{lo}, b, i}) = \begin{cases} 0 & \text{if } c_{b,i} \ge 0 \\ 15 c_{b,i} & \text{if } c_{b,i} < 0 \end{cases}, \qquad \max_{q_{\text{lo}} \in [0, 15]} (c_{b,i} q_{\text{lo}, b, i}) = \begin{cases} 15 c_{b,i} & \text{if } c_{b,i} \ge 0 \\ 0 & \text{if } c_{b,i} < 0 \end{cases}$$
Summing across block elements, taking the sign of $d_b$ into account:
$$\text{term}_{\text{neg}}(b) = 15 d_b \sum_{i: h_{b,i} < 0} h_{b,i}, \qquad \text{term}_{\text{pos}}(b) = 15 d_b \sum_{i: h_{b,i} > 0} h_{b,i}$$
$$\text{block\_min}(b) = \min(\text{term}_{\text{neg}}(b), \text{term}_{\text{pos}}(b)), \qquad \text{block\_max}(b) = \max(\text{term}_{\text{neg}}(b), \text{term}_{\text{pos}}(b))$$
Summing over all $B$ blocks yields the **exact infimum and supremum** on $[0, 15]^{32 \times B}$:
$$\text{lo}(\text{row}) = s_{\text{hi}}(\text{row}) + \sum_{b=0}^{B-1} \text{block\_min}(b), \qquad \text{hi}(\text{row}) = s_{\text{hi}}(\text{row}) + \sum_{b=0}^{B-1} \text{block\_max}(b)$$
The interval width is:
$$\text{hi}(\text{row}) - \text{lo}(\text{row}) = 15 \sum_{b=0}^{B-1} |d_b| \|h_b\|_1$$

#### Pruning and Argmax Exactness
- **Stage 1**: Read scales $d_b$ (2 B/block) and high nibbles $q_{\text{hi}}$ (16 B/block) for all $V$ rows ($18\text{ B/block} \implies 576\text{ B/row}$ on Qwen3.5, $504\text{ B/row}$ on Qwen2.5). Compute $s_{\text{hi}}$, $\text{lo}$, and $\text{hi}$ for all rows.
- **Pruning Threshold**:
  1. *Standard bound*: $m = \max_{r \in [0, V)} \text{lo}(r)$.
  2. *Candidate-seeded bound*: Evaluate dense logit for candidate row $k = \arg\max_r s_{\text{hi}}(r) \implies m_0 = \text{logit}(k)$.
- **Stage 2**: Any row with $\text{hi}(r) < m$ (or $< m_0$) is provably non-maximal and pruned. For surviving rows $\mathcal{S} = \{r \mid \text{hi}(r) \ge m\}$, read the low nibbles $q_{\text{lo}}$ (16 B/block $\implies 512\text{ B/row}$ on Qwen3.5, $448\text{ B/row}$ on Qwen2.5) and evaluate exact logits.
- **Exactness Invariant**: Since $\text{logit}(r) \le \text{hi}(r)$ unconditionally for all $r$, the global argmax row $r^*$ satisfies $\text{hi}(r^*) \ge \text{logit}(r^*) \ge m$, so $r^* \in \mathcal{S}$ is never pruned. Argmax over $\mathcal{S}$ is bit-exact to dense argmax.

---

### 8.2 Three-Stage Split and Max |h| Relaxation Variants

1. **Three-Stage Bit-Plane Split**:
   Codes are decomposed into 2-bit top ($q_{\text{top2}} = \lfloor q/64 \rfloor \in [-2, 1]$), 2-bit mid ($q_{\text{mid2}} \in [0, 3]$), and 4-bit low ($q_{\text{lo4}} \in [0, 15]$).
   - In Stage 1 (top 2 bits + scale = 10 B/block), the unobserved remainder $q - 64 q_{\text{top2}} \in [0, 63]$ spans 6 bits. The Stage 1 interval width expands by $63/15 = 4.2\times$ to $\approx 73.4\text{ logits}$.
   - Because $73.4\text{ logits}$ exceeds the vocabulary logit span, **$100.00\%$ of rows survive Stage 1** across all 832 positions. Stage 1 achieves zero pruning, while adding memory streaming overhead.
2. **Per-Block Max |h| Relaxation**:
   Bounding $\sum_{i \in b} q_{\text{lo}, i} h_i \le 15 \cdot 32 \cdot |d_b| \max_{i} |h_{b, i}| = 480 |d_b| \max_i |h_{b, i}|$ relaxes element-wise sign alignment.
   - The median interval width inflates from $17.48\text{ logits}$ to **$128.51\text{ logits}$** ($7.35\times$ looser).
   - At $128.51\text{ logits}$, **$100.00\%$ of rows survive** on all 832 positions ($0.00\%$ pruning).
   - *Confirmation*: The sign-split sum achieves the exact attainable extrema over the hypercube $[0, 15]^{32}$. Any $\max |h|$ relaxation destroys pruning selectivity.

---

### 8.3 Empirical Results: Qwen3.5-0.8B Head (832 Positions)

Evaluated across the 832 real decode positions from `h_states.npy` (normed final states $h = \text{rms\_norm}(x, \text{output\_norm})$). All 832 positions verified for exact argmax match ($832/832$, $100.00\%$):

| Pruning Configuration | Exact Match | Median Width (logits) | Survivors (Med) | Survivors (p90) | Survivors (Max) | Head Bytes Read (Med) | Frac of Dense (Med) | Frac of Dense (p90) |
|---|---|---|---|---|---|---|---|---|
| **Dense Baseline (Q8_0)** | $832 / 832$ | — | $248{,}320$ ($100\%$) | $248{,}320$ ($100\%$) | $248{,}320$ ($100\%$) | $270.17\text{ MB}$ | $100.00\%$ | $100.00\%$ |
| **Standard Bit-Plane ($m = \max \text{lo}$)** | **$832 / 832$** | $17.48$ | $26{,}236.5$ ($10.57\%$) | $105{,}716.3$ ($42.57\%$) | $230{,}811$ ($92.95\%$) | $156.47\text{ MB}$ | **$57.91\%$** | $72.98\%$ |
| **Candidate Seeded ($m_0 = \text{logit}(\arg\max s_{\text{hi}})$)** | **$832 / 832$** | $17.48$ | **$246.0$ ($0.099\%$)** | **$1{,}383.0$ ($0.557\%$)** | $10{,}068$ ($4.054\%$) | **$143.16\text{ MB}$** | **$52.99\%$** | **$53.20\%$** |
| **3-Stage Split (top 2b / mid 2b / low 4b)** | **$832 / 832$** | $73.42$ (S1) / $17.48$ (S2) | $248{,}320$ (S1) / $26{,}236$ (S2) | $248{,}320$ / $105{,}716$ | $248{,}320$ / $230{,}811$ | $156.47\text{ MB}$ | $57.91\%$ | $72.98\%$ |
| **Max $|h|$ Relaxation Variant** | **$832 / 832$** | $128.51$ | $248{,}320$ ($100.0\%$) | $248{,}320$ ($100.0\%$) | $248{,}320$ ($100.0\%$) | $270.17\text{ MB}$ | $100.00\%$ | $100.00\%$ |

#### Qwen3.5-0.8B Domain Breakdown (Standard $m = \max \text{lo}$):
| Domain | Positions $N$ | Median Width (logits) | Survivors Frac (Med) | Survivors Frac (p90) | Survivors Frac (Max) | Head Bytes Frac (Med) |
|---|---|---|---|---|---|---|
| `code` (`src/engine/attn.rs`) | $300$ | $17.45$ | $5.56\%$ ($13{,}804$ rows) | $37.40\%$ | $84.15\%$ | $55.56\%$ |
| `prose` (*Pride and Prejudice*) | $300$ | $17.90$ | $10.51\%$ ($26{,}095$ rows) | $34.43\%$ | $87.94\%$ | $57.89\%$ |
| `readme_gen` (Greedy generated) | $32$ | $17.09$ | $5.71\%$ ($14{,}169$ rows) | $40.57\%$ | $70.69\%$ | $55.63\%$ |
| `readme_prompt` (README prompt) | $200$ | $16.89$ | $19.14\%$ ($47{,}520$ rows) | $54.61\%$ | $92.95\%$ | $61.95\%$ |
| **Overall** | **832** | **17.48** | **10.57%** ($26{,}236$ rows) | **42.57%** | **92.95%** | **57.91%** |

---

### 8.4 Empirical Results: Qwen2.5-0.5B Head (800 Positions)

The tied LM head in `/home/socol/models/gguf/qwen05b-Q4_K_M.gguf` (`token_embd.weight`, shape $151{,}936 \times 896$, $B = 28$ blocks, $952\text{ B/row}$, dense $144.64\text{ MB}$) is also confirmed type `Q8_0` (tensor type 8). Evaluated on 800 real decode positions across the same domains:

| Pruning Configuration | Exact Match | Median Width (logits) | Survivors (Med) | Survivors (p90) | Survivors (Max) | Head Bytes Read (Med) | Frac of Dense (Med) | Frac of Dense (p90) |
|---|---|---|---|---|---|---|---|---|
| **Dense Baseline (Q8_0)** | $800 / 800$ | — | $151{,}936$ ($100\%$) | $151{,}936$ ($100\%$) | $151{,}936$ ($100\%$) | $144.64\text{ MB}$ | $100.00\%$ | $100.00\%$ |
| **Standard Bit-Plane ($m = \max \text{lo}$)** | **$800 / 800$** | $17.29$ | $29{,}349.0$ ($19.32\%$) | $80{,}509.2$ ($52.99\%$) | $140{,}292$ ($92.34\%$) | $89.72\text{ MB}$ | **$62.03\%$** | $77.88\%$ |
| **Candidate Seeded ($m_0 = \text{logit}(\arg\max s_{\text{hi}})$)** | **$800 / 800$** | $17.29$ | **$284.0$ ($0.187\%$)** | **$2{,}033.9$ ($1.339\%$)** | $12{,}473$ ($8.209\%$) | **$76.70\text{ MB}$** | **$53.03\%$** | **$53.57\%$** |

#### Qwen2.5-0.5B Domain Breakdown (Standard $m = \max \text{lo}$):
| Domain | Positions $N$ | Median Width (logits) | Survivors Frac (Med) | Survivors Frac (p90) | Survivors Frac (Max) | Head Bytes Frac (Med) |
|---|---|---|---|---|---|---|
| `code` | $300$ | $17.08$ | $11.45\%$ | $41.94\%$ | $90.05\%$ | $58.33\%$ |
| `prose` | $300$ | $18.22$ | $19.40\%$ | $48.66\%$ | $91.75\%$ | $62.07\%$ |
| `readme` | $200$ | $15.89$ | $29.82\%$ | $65.68\%$ | $92.34\%$ | $66.97\%$ |
| **Overall** | **800** | **17.29** | **19.32%** | **52.99%** | **92.34%** | **62.03%** |

---

### 8.5 Model-Grade Throughput and Memory Bandwidth Projections

Using Apeiron's measured DDR5 host memory bandwidth floor ($82\text{ GB/s}$) and measured baseline decode latency ($7.80\text{ ms/tok} \implies 128.2\text{ tok/s}$):
- Dense LM head read traffic: $270.17\text{ MB} \implies 3.30\text{ ms}$.
- Non-LM-head execution time (24 transformer layers): $7.80 - 3.30 = 4.50\text{ ms}$.

| Configuration | Head Bytes Read (Med) | Head Read Latency (82 GB/s) | Non-Head Latency | Projected Decode Latency | Projected Tok/s | Projected Speedup |
|---|---|---|---|---|---|---|
| **Dense Baseline** | $270.17\text{ MB}$ ($100.0\%$) | $3.30\text{ ms}$ | $4.50\text{ ms}$ | $7.80\text{ ms}$ | **$128.2\text{ tok/s}$** | $1.00\times$ |
| **Standard ($m = \max \text{lo}$)** | $156.47\text{ MB}$ ($57.9\%$) | $1.91\text{ ms}$ | $4.50\text{ ms}$ | $6.41\text{ ms}$ | **$156.0\text{ tok/s}$** | **$1.22\times$ (+21.7%)** |
| **Candidate Seeded ($m_0 = \text{logit}(\arg\max s_{\text{hi}})$)** | $143.16\text{ MB}$ ($53.0\%$) | $1.75\text{ ms}$ | $4.50\text{ ms}$ | $6.25\text{ ms}$ | **$160.0\text{ tok/s}$** | **$1.25\times$ (+24.8%)** |

*Grade*: `model` (projection based on measured 82 GB/s memory bandwidth floor, assuming ideal transposed/bit-plane tensor storage layout).

---

### 8.6 Hardware Reality and Storage Format Constraint

While the bit-plane bound passes the mathematical score-gap condition, its physical memory savings depend critically on tensor layout:

1. **Standard Interleaved GGUF `Q8_0` Layout**:
   In standard GGUF files, each 34-byte block contains 2 scale bytes followed immediately by 32 int8 code bytes. Each byte stores $(q_{\text{hi}} \ll 4) \mid q_{\text{lo}}$ together.
   Host CPU memory controllers transfer cache lines in 64-byte bursts. Reading only $q_{\text{hi}}$ from standard Q8_0 fetches the full 64-byte cache line containing $q_{\text{lo}}$, resulting in **$0.0\%$ DRAM bandwidth reduction** in unmodified GGUF format.
2. **Required Bit-Plane Split Layout**:
   To achieve the measured $42.1\%\text{–}47.0\%$ bandwidth reduction, the model must be converted offline into two contiguous streams:
   - Stream 1 (`head_hi.bin`, $143.03\text{ MB}$): scales $d_b$ ($15.89\text{ MB}$) + packed high nibbles $q_{\text{hi}}$ ($127.14\text{ MB}$). Read sequentially at 82 GB/s in Stage 1.
   - Stream 2 (`head_lo.bin`, $127.14\text{ MB}$): packed low nibbles $q_{\text{lo}}$.
3. **Scattered Gather Latency**:
   In Stage 2, surviving rows ($246$ rows for seeded $m_0$, or $26{,}236$ rows for standard $m$) require reading $512$ bytes per survivor from `head_lo.bin`. While the volume is small ($0.13\text{ MB}$ for $m_0$; $13.4\text{ MB}$ for standard), non-sequential scattered reads incur DDR5 DRAM page-open/close overheads and cache-line striding penalties that will reduce effective bandwidth below the 82 GB/s streaming peak.

---

### 8.7 Instrument Validation and Controls

1. **Exactness Invariant (Positive Control)**:
   For all $832$ positions on Qwen3.5-0.8B and all $800$ positions on Qwen2.5-0.5B across both pruning thresholds ($m = \max \text{lo}$ and $m_0 = \text{logit}(\arg\max s_{\text{hi}})$):
   $$\arg\max_{i \in \mathcal{S}} (w_i \cdot h) \equiv \arg\max_{i \in [0, V)} (w_i \cdot h)$$
   Total tests: $(2 \times 832) + (2 \times 800) = 3264$ checks; **$0$ mismatches** ($100.00\%$ bit-exact match).
2. **Seeded Defect (Negative Control)**:
   The mutation control (`/home/socol/oc-tmp/lmhead-t1/control.py`) mutates the upper bound by setting the slack factor to $0.0$ ($\text{hi}_{\text{mut}} = s_{\text{hi}}$, ignoring low-nibble interval expansion):
   - Sound upper bound ($\text{hi}(i) \ge \text{logit}(i)$): **$0 / 832$ violations**.
   - Sound argmax pruning preservation: **$0 / 832$ mismatches**.
   - Mutated upper bound ($\text{hi}_{\text{mut}} < \text{logit}$): **$583 / 832$ violations** detected ($70.07\%$).
   - Mutated candidate pruning ($\text{hi}_{\text{mut}} < m_0$ on true argmax): **$568 / 832$ mismatches** detected ($68.27\%$).
   - Negative control PASS: instrument catches unsound bound mutations with attributable red diagnostics.

---

### 8.8 Decision Line

1. **Does the bit-plane bound pass the score-gap condition?**
   **YES.** Unlike geometric outer-ball and PCA bounds—which produced $25.6\text{–}67.2\text{ logits}$ of slack and pruned $< 4\%$ of rows—the bit-plane bound achieves an interval width of **$17.48\text{ logits}$** (median).
   - Standard interval pruning ($m = \max \text{lo}$) skips **$89.43\%$ of rows** (median $26{,}236$ survivors), reducing LM head memory traffic from $270.2\text{ MB}$ to **$156.5\text{ MB}$** ($57.9\%$ of dense).
   - Candidate-seeded pruning ($m_0 = \text{logit}(\arg\max s_{\text{hi}})$) skips **$99.90\%$ of rows** (median $246$ survivors), reducing LM head traffic to **$143.2\text{ MB}$** ($53.0\%$ of dense).
2. **Projected throughput**:
   In the memory bandwidth-bound host CPU regime ($82\text{ GB/s}$ DDR5), this projects a throughput increase from $128.2\text{ tok/s}$ to **$156.0\text{–}160.0\text{ tok/s}$** ($1.22\times\text{–}1.25\times$ speedup).
3. **Engineering Precondition**:
   This speedup **cannot be achieved in-place on standard GGUF `Q8_0` files** because interleaved byte packing forces 64-byte DRAM cache-line loads. It requires an offline conversion to a split 2-stream layout (`Q8_0_BITPLANE`), pairing contiguous streaming of high nibbles with a vectorized Stage 2 gather.

## Licence (proved)

`proof/corpus/Numeric/BitplaneArgmax.lean` (`70ae501c`), registered in `AxiomAudit.lean` (`lake build AxiomAudit` exit 0, 9074 jobs):
`nibble_split`, `nibble_hi_range` (codes in [−128, 127] split as 16·q_hi + q_lo, q_hi ∈ [−8, 7], q_lo ∈ [0, 15]);
`lo_interval_sound` (hiPart + loLower ≤ dot ≤ hiPart + loUpper, both scale signs); `bitplane_prune_sound`,
`survivors_nonempty`, `bitplane_argmax_exact` (the lowest-index argmax over {r | dot k ≤ hi r} equals the global one,
through `PrunedArgmax.lowestMaxOn`); witnesses on a 3-row instance built from the module's own `hiPart`/`loUpper`;
`lo_upper_needed` (dropping the low-nibble slack is unsound there). Controls: range 15 → 7 (worker) fails at
`lo_interval_sound`; strict survivor inequality (lead, `/home/socol/oc-tmp/lead-bitplane-ctl/`) fails at
`bitplane_prune_sound` and `survivors_nonempty`. Not covered: the f16 → Q32.32 scale conversion (DequantQ8_0's
refinement) and the Rust kernel.
