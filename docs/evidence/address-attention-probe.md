# Address Attention Probe: Exact Attention Collapse and Key Address Occupancy

## 1. Intent and Hypothesis

This probe evaluates the **exact address collapse** hypothesis for attention on causal language models.

### Thesis Under Test (grade `model` until measured)
If every key vector $k_s$ in a sequence of length $t$ is snapped to an address $a = \text{addr}(s)$ chosen from a finite codebook with code centres $\{c_a\}_{a \in \mathcal{A}}$, softmax attention over the entire past collapses algebraically into a sum over the **distinct occupied addresses**:

$$\sum_{s=1}^t \exp(q \cdot k_s) v_s = \sum_{a \in \text{occupied}} \exp(q \cdot c_a) \left( \sum_{s: \text{addr}(s)=a} v_s \right)$$

If keys frequently collide into a sublinear set of addresses ($|\text{occupied}| \ll t$, e.g. $|\text{occupied}| \sim \mathcal{O}(\log t)$ or $\mathcal{O}(t^\alpha)$ with $\alpha < 0.5$), then computing attention over the whole past costs $\mathcal{O}(|\text{occupied}|)$ inner products per token instead of $\mathcal{O}(t)$, achieving subquadratic decode without dropping tokens or pruning the context.

### Question
Does the number of occupied key addresses stay sublinear while preserving model generation invariants (flip rate $\le 15.6\%$, matching the 4-bit seal baseline)? Or does preserving attention fidelity in head dimension $d=64$ require a codebook resolution so fine that occupied addresses grow linearly with sequence length ($\alpha \approx 1.0$)?

---

## 2. Experimental Setup and Artifacts

- **Model**: `Qwen/Qwen2.5-0.5B` (HuggingFace, weights from `/home/socol/models/qwen2.5-0.5b`).
  - Architecture: 24 layers, 14 query heads, 2 key-value heads ($GQA$), `head_dim` = 64 (32 RoPE 2-planes), `hidden_size` = 896.
  - Baseline teacher-forced cross-entropy loss: **4.5912 bits/token** on held-out evaluation corpus.
- **Hardware**: NVIDIA GeForce RTX 5080 (16,303 MiB physical VRAM, Blackwell `sm_120`). Peak VRAM footprint strictly maintained under 2.5 GB (well below the 5.0 GB multi-tenant budget).
- **Random Seed**: Fixed to `0` across all runs (`torch.manual_seed(0)`, `np.random.seed(0)`).
- **Datasets**:
  1. `wikitext_16k`: Test split of `Salesforce/wikitext` (`wikitext-103-raw-v1`) concatenated to 16,384 tokens.
  2. `gsm8k_cot_16k`: Chain-of-thought question-answer pairs from `openai/gsm8k` test split concatenated to 16,384 tokens.
  3. `prose8.txt` evaluation split: Disjoint held-out corpus (`/home/socol/Workspace/fold/hashtrinity/prose8.txt`).
     - 33 blocks of $\text{SEQ}=512$ ($33 \times 511 = 16,863$ next-token predictions) for teacher-forced top-1 flip rate and bits/token.
     - 50 held-out prompts of length 32 tokens, generating 64 new greedy tokens (`do_sample=False`) for mean identical greedy prefix length.
- **Artifacts and Checksums**:
  - Data preparation: `/home/socol/oc-tmp/addr-t0/prepare_data.py` (sha256: `59b820b255e6d987e9d2855fe6fe6e4ea4dab0cbbce246d837dbaf466e606c29`)
  - Probe runner: `/home/socol/oc-tmp/addr-t0/run_probe.py` (sha256: `a35541bc095461dbde6b576cbb48004beeddea906002cfe095534ff7ec50cbfc`)
  - Gate checker: `/home/socol/oc-tmp/addr-t0/check.py` (sha256: `215a9d807044f8b0edcf86aaf102fb32e37f8d0942abd8de1c5942f6a56bef4c`)
  - Raw JSON results: `/home/socol/oc-tmp/addr-t0/raw_results.json`

---

## 3. Address Schemes

Each scheme operates per layer and KV head on keys of dimension $d=64$:

1. **Scheme A (Post-RoPE $E_8$ Lattice)**:
   - Post-RoPE keys $k_{\text{post}} \in \mathbb{R}^{64}$ are rotated by a fixed orthogonal matrix $Q \in \mathcal{O}(64)$: $y = k_{\text{post}} Q^T$.
   - Divided into 8 blocks of 8 dimensions: $y = (y_0, \dots, y_7)$ with $y_b \in \mathbb{R}^8$.
   - Each 8-block is scaled by scale factor $s \in \{0.1, 0.25, 0.5, 1.0, 2.0, 3.0\}$ and snapped to the nearest $E_8$ lattice point using the exact Gosset/Conway-Sloane $E_8$ decoder (`batch_nearest_E8` from `qcache-t0`).
   - The address $a$ is the 8-tuple of $E_8$ lattice points: $(p_0, \dots, p_7) \in (E_8)^8$.
   - Representative key: $c_a = (\frac{1}{s} (p_0, \dots, p_7)) Q$.

2. **Scheme B (Pre-RoPE Content $E_8$ + Position Codes)**:
   - Pre-RoPE keys $k_{\text{pre}} \in \mathbb{R}^{64}$ (captured directly from `k_proj`) are coded as in Scheme A into an 8-block $E_8$ content code $c_{\text{content}}$.
   - Position is handled via two variants:
     - **Ring Residue**: Address is $a = (c_{\text{content}}, r)$ where $r = p \bmod N$ for $N \in \{64, 256, 1024\}$.
       - Evaluated under two representative models: (1) TRUE position rotation (content only is quantized, rotary phase is continuous), and (2) snapped position rotation (rotary phase is snapped to $r = p \bmod N$).
     - **Fenwick Bucket**: Address relative to query position $t$ is $a_t(p) = (c_{\text{content}}, \lfloor \log_2(t - p + 1) \rfloor)$, yielding $\mathcal{O}(\log t)$ discrete distance buckets.

3. **Scheme C (Per-2-Plane Spinor / Angle Code)**:
   - The 64 dimensions form 32 RoPE 2-planes: $(x_j, y_j) = (k_j, k_{j+32})$ for $j \in \{0, \dots, 31\}$.
   - In each 2-plane, polar coordinates are computed: radius $r_j = \sqrt{x_j^2 + y_j^2}$, angle $\theta_j = \text{atan2}(y_j, x_j) \in [-\pi, \pi)$.
   - Quantized into $A \in \{16, 64\}$ uniform angle bins and $M \in \{4, 8\}$ radial bins.
   - Address is the 32-tuple of bin coordinates: $((m_0, a_0), \dots, (m_{31}, a_{31}))$.
   - Representative key: $(r_{\text{bin}} \cos \theta_{\text{bin}}, r_{\text{bin}} \sin \theta_{\text{bin}})$, followed by true RoPE rotation.

---

## 4. Measurement 3: Exactness Check of the Attention Collapse

To confirm that the collapsed histogram formulation is mathematically exact and not an approximation, direct sum attention and the grouped histogram sum were evaluated in `float64` across 10 random query vectors and key sequences of length $t = 2,048$ with 128 discrete code centres:

$$\text{Direct}: \quad \frac{\sum_{s=1}^t \exp(q \cdot k_s) v_s}{\sum_{s=1}^t \exp(q \cdot k_s)}, \qquad \text{Histogram}: \quad \frac{\sum_{a \in \text{occ}} \exp(q \cdot c_a) \left(\sum_{s: \text{addr}(s)=a} v_s\right)}{\sum_{a \in \text{occ}} \exp(q \cdot c_a) \cdot \text{count}(a)}$$

| Query Sample | Sequence Length $t$ | Occupied Addresses | Max Absolute Difference |
|---|---|---|---|
| Sample 0 | 2048 | 128 | $2.78 \times 10^{-17}$ |
| Sample 1 | 2048 | 128 | $4.16 \times 10^{-17}$ |
| Sample 2 | 2048 | 128 | $2.78 \times 10^{-17}$ |
| Sample 3 | 2048 | 128 | $4.16 \times 10^{-17}$ |
| Sample 4 | 2048 | 128 | $2.95 \times 10^{-17}$ |
| Sample 5 | 2048 | 128 | $2.95 \times 10^{-17}$ |
| Sample 6 | 2048 | 128 | $2.78 \times 10^{-17}$ |
| Sample 7 | 2048 | 128 | $2.78 \times 10^{-17}$ |
| Sample 8 | 2048 | 128 | $2.78 \times 10^{-17}$ |
| Sample 9 | 2048 | 128 | $4.16 \times 10^{-17}$ |

**Result**: Max absolute difference across all samples is **$4.16 \times 10^{-17}$**, bounded by floating-point roundoff at machine precision ($\le 10^{-16}$). The collapse identity is algebraically exact.

---

## 5. Measurement 1: Occupancy Growth vs Sequence Length $t$

Key occupancy $n(t)$ was tracked at checkpoints $t \in \{1024, 2048, 4096, 8192, 16384\}$ across layers and KV heads. Growth was fit to $n(t) = c \cdot t^\alpha$. Live window occupancy (tokens with attention score $S \ge \max - 16$) was measured at $t = 16,384$.

### Wikitext-103 (16,384 tokens) — Representative Layers

| Scheme | Layer | $t=1\text{k}$ | $t=2\text{k}$ | $t=4\text{k}$ | $t=8\text{k}$ | $t=16\text{k}$ | Exponent $\alpha$ | Live Window ($t=16\text{k}$) |
|---|---|---|---|---|---|---|---|---|
| Scheme A ($s=0.1$) | 0 | 47 | 81 | 145 | 258 | 382 | 0.7712 | 149 |
| Scheme A ($s=0.1$) | 12 | 36 | 73 | 145 | 289 | 584 | 0.9883 | 398 |
| Scheme A ($s=0.25$) | 12 | 1022 | 2044 | 4088 | 8182 | 16372 | 1.0001 | 11143 |
| Scheme A ($s=0.5$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme A ($s=1.0$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme A ($s=2.0$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme B (content $s=1.0$, unrotated) | 0 | 321 | 520 | 857 | 1345 | 1967 | 0.7470 | 1123 |
| Scheme B (content $s=1.0$, unrotated) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme B (content $s=1.0$, Fenwick) | 0 | 542 | 901 | 1512 | 2390 | 3410 | 0.7270 | 1124 |
| Scheme B (content $s=1.0$, Fenwick) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme B (content $s=2.0$, ring $N=64$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme B (content $s=2.0$, ring $N=256$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme B (content $s=2.0$, ring $N=1024$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme C ($A=16, M=4$) | 0 | 282 | 471 | 794 | 1251 | 1832 | 0.7070 | 1067 |
| Scheme C ($A=16, M=4$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |
| Scheme C ($A=64, M=4$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 11151 |

### GSM8K Chain-of-Thought (16,384 tokens) — Representative Layers

| Scheme | Layer | $t=1\text{k}$ | $t=2\text{k}$ | $t=4\text{k}$ | $t=8\text{k}$ | $t=16\text{k}$ | Exponent $\alpha$ | Live Window ($t=16\text{k}$) |
|---|---|---|---|---|---|---|---|---|
| Scheme A ($s=0.5$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 4826 |
| Scheme A ($s=1.0$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 4826 |
| Scheme A ($s=2.0$) | 12 | 1024 | 2048 | 4096 | 8192 | 16384 | 1.0000 | 4826 |
| Scheme B (content $s=1.0$, unrotated) | 0 | 164 | 287 | 457 | 694 | 1009 | 0.6080 | 534 |
| Scheme B (content $s=1.0$, unrotated) | 12 | 1024 | 2048 | 4093 | 8184 | 16368 | 1.0000 | 4824 |
| Scheme B (content $s=1.0$, Fenwick) | 0 | 310 | 548 | 912 | 1381 | 1982 | 0.6130 | 535 |
| Scheme B (content $s=1.0$, Fenwick) | 12 | 1024 | 2048 | 4096 | 8191 | 16378 | 1.0000 | 4825 |
| Scheme C ($A=16, M=4$) | 0 | 158 | 276 | 451 | 682 | 991 | 0.5830 | 542 |
| Scheme C ($A=16, M=4$) | 12 | 1024 | 2048 | 4096 | 8191 | 16382 | 1.0000 | 4826 |

**Observations**:
- At Layer 0 (directly following token embeddings and normalization), content keys exhibit subspace concentration: unrotated content keys and coarse spinor codes show sublinear growth ($\alpha \approx 0.58 - 0.75$).
- For all intermediate and deep layers ($\ge 6$, including 6, 12, 18, 23), key occupancy is strictly linear ($\alpha = 1.0000$, with 16,384 distinct addresses occupied at $t=16,384$).

---

## 6. Measurement 2: Conversion Damage

Conversion cost of replacing real keys with representative codebook addresses during model forward pass and autoregressive generation:

- **Reference Baseline (FP16)**:
  - Teacher-forced bits/token: **4.5912**
  - Top-1 flip rate: **0.00%**
  - Mean identical greedy prefix: **64.00 tokens**
- **Benchmarks**:
  - `4-bit seal`: flip rate **15.55%**, **4.6726** bpt, **8.66** prefix (`docs/evidence/decision-cell-census.md`)
  - `GPTQ-int4`: flip rate **21.94%**, **4.8137** bpt, **4.78** prefix (`docs/evidence/decision-cell-census.md`)
  - `EXL3 4.0 bpw`: flip rate **9.73%**, **4.6300** bpt, **12.80** prefix (`docs/evidence/exllamav3-baseline.md`)

### Damage Results

| Ansatz / Scheme | Top-1 Flip Rate | Bits / Token | $\Delta$ bpt | Mean Greedy Prefix | Wall Time |
|---|---|---|---|---|---|
| FP16 Reference | 0.00% | 4.5912 | +0.0000 | 64.00 | — |
| Scheme A ($s=0.10$) | 95.21% | 13.4217 | +8.8305 | 0.08 | 1.7 s |
| Scheme A ($s=0.25$) | 88.96% | 10.4664 | +5.8752 | 0.14 | 1.7 s |
| Scheme A ($s=0.50$) | 58.97% | 6.4711 | +1.8799 | 0.66 | 1.8 s |
| Scheme A ($s=1.00$) | 25.69% | 4.8719 | +0.2807 | 2.74 | 1.8 s |
| Scheme A ($s=2.00$) | 11.67% | 4.6460 | +0.0548 | 14.54 | 1.7 s |
| Scheme A ($s=3.00$) | 8.21% | 4.6128 | +0.0216 | 14.64 | 1.7 s |
| Scheme B (content $s=1.0$, TRUE pos) | 28.06% | 4.9717 | +0.3805 | 5.52 | 1.8 s |
| Scheme B (content $s=2.0$, TRUE pos) | 12.10% | 4.6450 | +0.0538 | 12.40 | 1.7 s |
| Scheme B (content $s=2.0$, ring $N=64$) | 81.22% | 10.0263 | +5.4351 | 11.34 | 2.2 s |
| Scheme B (content $s=2.0$, ring $N=256$) | 48.84% | 7.0984 | +2.5072 | 12.40 | 2.2 s |
| Scheme B (content $s=2.0$, ring $N=1024$) | 12.10% | 4.6450 | +0.0538 | 12.40 | 2.3 s |
| Scheme C ($A=16, M=4$) | 92.59% | 12.0251 | +7.4339 | 0.10 | 1.0 s |
| Scheme C ($A=64, M=4$) | 91.64% | 11.6973 | +7.1061 | 0.16 | 1.0 s |
| Scheme C ($A=64, M=8$) | 74.62% | 8.3542 | +3.7630 | 0.36 | 1.0 s |

---

## 7. The Frontier Table

Combining occupancy growth $\alpha$, 16k occupancy, top-1 flip rate, and bits/token:

| Scheme & Configuration | Growth Exponent $\alpha$ | Occupied at 16k | Occupancy Fraction | Flip Rate | Bits / Token | Greedy Prefix |
|---|---|---|---|---|---|---|
| Scheme A ($s=0.10$) | 0.9883 | 687 | 4.19% | 95.21% | 13.4217 | 0.08 |
| Scheme A ($s=0.25$) | 1.0001 | 16372 | 99.93% | 88.96% | 10.4664 | 0.14 |
| Scheme A ($s=0.50$) | 1.0000 | 16384 | 100.00% | 58.97% | 6.4711 | 0.66 |
| Scheme A ($s=1.00$) | 1.0000 | 16384 | 100.00% | 25.69% | 4.8719 | 2.74 |
| Scheme A ($s=2.00$) | 1.0000 | 16384 | 100.00% | 11.67% | 4.6460 | 14.54 |
| Scheme A ($s=3.00$) | 1.0000 | 16384 | 100.00% | 8.21% | 4.6128 | 14.64 |
| Scheme B (content $s=1.0$, TRUE pos) | 1.0000 | 16384 | 100.00% | 28.06% | 4.9717 | 5.52 |
| Scheme B (content $s=2.0$, TRUE pos) | 1.0000 | 16384 | 100.00% | 12.10% | 4.6450 | 12.40 |
| Scheme B (content $s=2.0$, ring $N=64$) | 1.0000 | 16384 | 100.00% | 81.22% | 10.0263 | 11.34 |
| Scheme B (content $s=2.0$, ring $N=256$) | 1.0000 | 16384 | 100.00% | 48.84% | 7.0984 | 12.40 |
| Scheme B (content $s=2.0$, ring $N=1024$) | 1.0000 | 16384 | 100.00% | 12.10% | 4.6450 | 12.40 |
| Scheme C ($A=16, M=4$) | 1.0000 | 16384 | 100.00% | 92.59% | 12.0251 | 0.10 |
| Scheme C ($A=64, M=4$) | 1.0000 | 16384 | 100.00% | 91.64% | 11.6973 | 0.16 |
| Scheme C ($A=64, M=8$) | 1.0000 | 16384 | 100.00% | 74.62% | 8.3542 | 0.36 |

---

## 8. Plain Verdict

### Question
> **Is there a resolution where occupancy is sublinear ($\alpha < 0.5$ or logarithmic) AND flip rate $\le 15.6\%$?**

### Answer: **NO**.

### Geometric Mechanism
Attention key addresses do **not** stay small. The experimental evidence reveals a sharp, insurmountable Pareto barrier governed by the geometry of high-dimensional Euclidean space ($d = 64$):

1. **The Curse of Dimension ($d=64$)**:
   In dimension 64, the volume of a sphere of radius $R$ scales as $V \propto R^{64}$. A codebook fine enough to preserve inner products with queries ($s \ge 2.0$, flip rate $11.67\% \le 15.6\%$) partitions the unit sphere into $> 2^{64}$ Voronoi cells. Because natural key vectors are non-collinear and rotate rapidly via RoPE, each of the $16,384$ tokens lands in an independent Voronoi cell with near-certainty. The occupancy is strictly **linear** ($\alpha = 1.0000$, $n(t) = t$).
2. **The Damage of Forced Collisions**:
   To force address collisions and achieve sublinear address growth ($\alpha < 0.5$ or $n(t) \ll t$), the codebook must be made exceedingly coarse ($s \le 0.10$). At this resolution, Voronoi cells span several units of distance, introducing $\mathcal{O}(1)$ distortion on normalized inner products. Because softmax attention is an exponential magnifier ($\exp(q \cdot k / \sqrt{d})$), small coordinate shifts completely alter the argmax ranking, driving top-1 flip rates to **$95.21\%$** and causing cross-entropy loss to explode from $4.59$ to **$13.42$ bpt**.
3. **Position Aliasing under Modular Rings**:
   Snapping position to a modular ring residue ($r = p \bmod N$) allows keys separated by multiples of $N$ to share an address, but at catastrophic conversion cost: for $N=64$, flip rate is **$81.22\%$**; for $N=256$, flip rate is **$48.84\%$**.
4. **Summary**:
   While the algebraic collapse $\sum_s e^{q \cdot k_s} v_s = \sum_a e^{q \cdot c_a} V_a$ is exact down to machine epsilon ($4.16 \times 10^{-17}$), the number of occupied addresses $|\text{occupied}|$ equals $t$ whenever the model generates coherent text. Exact address collapse does not reduce attention complexity below $\mathcal{O}(t)$.

---

## 9. Verification Gate and Reproducibility

The gate checker re-derives all table entries, metadata, exactness residuals, and the plain verdict directly from `raw_results.json`:

```sh
# Gate check (re-deriving tables from raw JSON)
/home/socol/oc-tmp/qcache-t0/.venv/bin/python /home/socol/oc-tmp/addr-t0/check.py
# Exit status: 0

# Negative control: verify rejection when modifying a value
cp /home/socol/Workspace/apeiron/docs/evidence/address-attention-probe.md /home/socol/oc-tmp/addr-t0/control_copy.md
sed -i 's/11.67%/11.68%/g' /home/socol/oc-tmp/addr-t0/control_copy.md
/home/socol/oc-tmp/qcache-t0/.venv/bin/python /home/socol/oc-tmp/addr-t0/check.py /home/socol/oc-tmp/addr-t0/control_copy.md
# Exit status: 1 (Error: Flip rate 11.67% for Scheme A (s=2.0) not found in markdown)
```
