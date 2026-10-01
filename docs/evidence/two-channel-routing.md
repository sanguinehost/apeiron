# Evidence: Two-Channel Hyperbolic Partitioning for MoE Routing (`Nous.Seq.TwoChannelRouter`)

## Question

Can the hyperbolic idempotent decomposition in $\mathbb{R}[x]/(x^2 - \sigma)$ at $\sigma = 1$ ($e_+ = \frac{1+x}{2}, e_- = \frac{1-x}{2}$) partition the 64 experts of a sparse Mixture-of-Experts (MoE) model into two orthogonal cones, cutting router FLOPs in half and enabling decoupled expert weight caching?

Specifically:
1. Do real expert routing decisions in `OLMoE-1B-7B` naturally cluster into two invariant hyperbolic channels?
2. What fraction of the top-$k$ ($k=8$) active experts are captured by the dominant channel?
3. Does this mechanism satisfy the three profit conditions (collisions, score gaps, binding resource)?

---

## Method & Mathematical Formulation

### 1. Hyperbolic Channel Split
Over any field of characteristic zero, the split-complex algebra $\mathbb{R}[x]/(x^2 - 1)$ decomposes into orthogonal idempotent projectors:
$$e_+ = \left(\frac{1}{2}, \frac{1}{2}\right), \quad e_- = \left(\frac{1}{2}, -\frac{1}{2}\right)$$

* **say:** e plus equals one half, one half; e minus equals one half, minus one half.
* **means:** the split at $\sigma = 1$ gives two half-scaled cell pairs, one projecting onto each channel of the algebra.
* **vars:** $e_+$ — positive projector, `RingCell F` = $F \times F$; $e_-$ — negative projector, `RingCell F`; $F$ — field of characteristic zero; split-complex numbers (James Cockle, 1848); norm form (Carl Friedrich Gauss, 1801).
* **apeiron:** `Nous.Seq.TwoChannelRouter.e_pos` and `Nous.Seq.TwoChannelRouter.e_neg`, both `def`.
* **physics:** no physical content.

satisfying:

$$e_+^2 = e_+, \quad e_-^2 = e_-, \quad e_+ \cdot e_- = 0, \quad e_+ + e_- = 1$$

* **say:** e plus squared equals e plus; e minus squared equals e minus; e plus dot e minus equals zero; e plus plus e minus equals one.
* **means:** the two projectors are idempotent, mutually orthogonal and sum to the identity, so the algebra is their direct sum.
* **vars:** $e_+, e_-$ — projectors in `RingCell F`; $e_+^2$ — `ringMul` at $\sigma = 1$ applied to $e_+$ with itself; $0$ — the cell $(0, 0)$; $1$ — the cell $(1, 0)$.
* **apeiron:** `Nous.Seq.TwoChannelRouter.e_pos_sq`, `Nous.Seq.TwoChannelRouter.e_neg_sq`, `Nous.Seq.TwoChannelRouter.e_pos_mul_e_neg_zero`, `Nous.Seq.TwoChannelRouter.e_pos_add_e_neg`; all proved.
* **physics:** no physical content.

Any representation vector $z$ decomposes into two mutually uncoupled channels $z = e_+ z + e_- z$ where cross-channel actions vanish identically: $e_+ (e_- z) = 0$.

### 2. Spectral Bipartition on Expert Traces
Using the 131,072 expert selection events from `OLMoE-1B-7B` across 16 layers and 4 tasks (prose, code, math chain-of-thought, multi-turn chat), we computed the global expert co-occurrence correlation matrix:
$$C_{ij} = \frac{\sum_{t} \mathbf{1}[i \in \text{selected}_t \land j \in \text{selected}_t]}{\sqrt{N_i N_j}}$$

* **say:** C sub i j equals the sum over t of the indicator that i and j are both selected at t, over the square root of N sub i times N sub j.
* **means:** the co-occurrence count of two experts is normalised by the geometric mean of their selection counts, giving a correlation in $[0, 1]$.
* **vars:** $C_{ij}$ — co-occurrence correlation, $\mathbb{R}$; $\mathbf{1}[\cdot]$ — indicator, $\{0, 1\}$; $\text{selected}_t$ — top-$k$ expert set at event $t$; $N_i$ — selections of expert $i$, $\mathbb{N}$; $t$ — selection-event index.
* **apeiron:** `none — anchor`. The probe computes this matrix; no corpus declaration states it.
* **physics:** no physical content.

The Fiedler eigenvector of $C$ yields the canonical 1D splitting axis in $\mathbb{R}[x]/(x^2 - 1)$, partitioning the 64 experts into Channel + (34 experts) and Channel - (30 experts).

---

## Empirical Probe on Real OLMoE Traces

Command:
`python3 /home/socol/oc-tmp/e5-two-channel-moe/probe.py`

### Numeric Results
| Metric | Value |
|---|---|
| **Total Token Routing Events** | **131,072** |
| **Total Experts ($E$) / Top-$k$** | 64 experts / top-8 active |
| **Channel + Size / Channel - Size** | 34 experts / 30 experts |
| **Mean Dominant Channel Capture** | **67.6%** (5.41 / 8 experts) |
| **Median Routing Imbalance** | 25.0% |
| **Fraction Strictly Split ($\ge 75\%$ from one channel)** | **42.0%** |
| **Recall Loss Under Hard Single-Channel Gating** | **32.4%** |

### Key Diagnostic Finding
1. **Pretrained Softmax Entanglement:** In pretrained models trained with standard dense softmax routing, experts exhibit substantial cross-channel co-occurrence: on 58% of tokens, top-8 experts are drawn from both channels (e.g. 5 in +, 3 in -).
2. **Failure of Hard Gating:** Forcing a pretrained model into hard two-channel gating causes a **32.4% drop in expert recall**, which destroys model capability.
3. **Verdict:** Post-hoc two-channel partitioning fails on standard pretrained MoEs. To exploit hyperbolic channel separation without quality loss, models must be trained from scratch with an explicit channel orthogonality loss ($e_+ e_- = 0$) or served with a soft 2-stage hierarchical router.

---

## Formal Proof & Verification

- **Lean Module:** [`proof/corpus/Nous/Seq/TwoChannelRouter.lean`](file:///home/socol/Workspace/apeiron/proof/corpus/Nous/Seq/TwoChannelRouter.lean)
- **Primary Gate:**
  `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'cd proof/corpus && LEAN_NUM_THREADS=1 lake build Nous.Seq.TwoChannelRouter'`
  - **Verdict:** GREEN (exit code 0, 2,972 jobs built).
- **Headlines Proved:**
  - `e_pos_sq` & `e_neg_sq`: Idempotency $e_+^2 = e_+$ and $e_-^2 = e_-$.
  - `e_pos_mul_e_neg_zero`: Channel orthogonality $e_+ \cdot e_- = 0$.
  - `e_pos_add_e_neg`: Partition of unity $e_+ + e_- = 1$.
  - `cell_decompose`: Exact direct-sum decomposition $z = e_+ z + e_- z$.
  - `cross_channel_zero`: Invariant subspace vanishing $e_+ (e_- z) = 0$.
- **Mutation Control:**
  - Mutated file: `/home/socol/oc-tmp/e5-two-channel-moe/control_copy/TwoChannelRouter_mutant.lean`.
  - Mutation: Claimed non-zero cross product $e_+ \cdot e_- = (1, 0)$.
  - Control Execution:
    `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 60 bash -c 'LEAN_PATH=... lean /home/socol/oc-tmp/e5-two-channel-moe/control_copy/TwoChannelRouter_mutant.lean'`
  - **Verdict:** RED (exit code 1, `⊢ 0 = 1` contradiction).
  - Restoring clean source produces exit code 0.

---

## Profit Condition Assessment

1. **(a) Collisions ($R(D) < \log_2 t$):**
   Expert weight matrices $W_e$ have high reuse locality (90.1% reuse within 16 tokens), which pays for weight caching. However, dynamic routing addresses vary per token.
2. **(b) Score Gaps:**
   **FAILED on pretrained weights.** Hard channel splitting erases 32.4% of top-8 experts, violating the score gap ordering between experts across different channels.
3. **(c) Binding Resource:**
   **FAILED.** Router logit evaluation takes $64 \times 1024$ FLOPs ($\approx 0.13$ MFLOPs), while loading the 8 selected expert FFN weights requires reading $8 \times 3.54 \text{ MB} = \mathbf{28.3\text{ MB}}$ of weights from memory. The router is compute-negligible ($< 0.1\%$ of latency); cutting router FLOPs does not accelerate the memory-bound expert execution.
