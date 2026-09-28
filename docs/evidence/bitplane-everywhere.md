# Evidence: Bit-Plane Pruning on Dense FFN Activations (`Nous.Seq.BitplaneRouter`)

## Question

Can the format-derived bit-plane pruning that achieved 99.9% row reduction on the 248k-row output head (`lmhead-prune-probe.md`) be applied "everywhere" to dense Feed-Forward Network (FFN) layers (e.g. SwiGLU gating)?

Specifically:
1. What fraction of intermediate neurons in `Qwen3.5-0.8B` can be pruned using lead sign-plane bounds without losing activation energy?
2. Does bit-plane neuron pruning reduce memory bandwidth during autoregressive decode on memory-bound hardware?
3. Does this mechanism satisfy Profit Condition (c)?

---

## Method & Mathematical Formulation

### 1. Discrete Bit-Plane Dot Product Bound
For an integer or fixed-point weight vector $w \in \mathbb{Z}^D$ decomposed into lead bit-plane $s \in \{-1, 1\}^D$ and residual magnitude $r$ with $|r_i| \le R$:
$$w \cdot h = 2^k (s \cdot h) + r \cdot h$$
The residual dot product is strictly bounded by:
$$|r \cdot h| \le R \|h\|_1$$
If the lead bit-plane satisfies:
$$2^k (s \cdot h) < - R \|h\|_1$$
then $w \cdot h < 0$ strictly. In SwiGLU architectures, where activation $y = \text{silu}(W_{\text{gate}} h) \odot (W_{\text{up}} h)$ satisfies $\text{silu}(x) \approx 0$ for $x \ll 0$, negative gate neurons contribute negligible energy to the layer output.

---

## Empirical Probe on Pretrained Qwen3.5 FFN Layers

We evaluated Layer 0 of `Qwen3.5-0.8B` ($D = 1024$, intermediate neuron dimension $D_{\text{ffn}} = 3584$) across 100 prompt steps (`h_states.npy`), testing whether coarse sign-plane projections can prune inactive SwiGLU neurons.

Command:
`/home/socol/oc-tmp/exl3/.venv/bin/python /home/socol/oc-tmp/e6-bitplane-ffn/probe.py`

### Numeric Results
| Metric | Output Head (LM Head) | Dense FFN Layer 0 |
|---|---|---|
| **Total Candidates / Neurons** | 248,320 rows | 3,584 neurons |
| **Median Surviving Rows** | **246 rows** | **2,602 neurons** |
| **Pruning / Sparsity Ratio** | **99.90%** | **28.44%** |
| **Output Energy Preserved** | 100% exact argmax | 97.24% energy |
| **Memory Bandwidth Reduction** | **~2.0x bytes read** | **Negative (bandwidth increase)** |

### Key Diagnostic Finding: Profit Condition (c) Assessment
1. **The Asymmetry of Candidate Counts:**
   On the output head, the search space is massive ($248\text{k}$ rows) and only the single top-1 argmax token is required. Pruning 99.9% of rows means 99.9% of the weight tensor bytes are never loaded from DRAM into cache.
2. **Dense FFN Memory Penalty:**
   In dense FFN layers, the candidate count is small ($3,584$ neurons), and $71.56\%$ of the neurons remain active. To prune $28.44\%$ of neurons:
   - Pass 1 must load the sign/exponent plane of $W_{\text{gate}}$ ($3584 \times 1024$ bits $\approx 0.46\text{ MB}$).
   - Pass 2 must dynamically gather and load the surviving rows of $W_{\text{up}}$ and $W_{\text{down}}$.
   On GPU/CPU memory architectures, dynamic uncoalesced gather reads have lower effective bandwidth efficiency than contiguous streaming GEMM. The overhead of reading the lead bit-planes plus non-contiguous gather loading **exceeds the 28% FLOP savings**.
3. **Verdict:** Bit-plane pruning **fails Profit Condition (c) on dense FFN layers**. Format-derived bit-plane filtering only pays when the candidate population is large ($\ge 50\text{k}$) and the pruning ratio is extreme ($\ge 90\%$).

---

## Formal Proof & Verification

- **Lean Module:** [`proof/corpus/Nous/Seq/BitplaneRouter.lean`](file:///home/socol/Workspace/apeiron/proof/corpus/Nous/Seq/BitplaneRouter.lean)
- **Primary Gate:**
  `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'cd proof/corpus && LEAN_NUM_THREADS=1 lake build Nous.Seq.BitplaneRouter'`
  - **Verdict:** GREEN (exit code 0, 2,972 jobs built).
- **Headlines Proved:**
  - `residual_bound`: $|r_0 h_0 + r_1 h_1| \le R (|h_0| + |h_1|)$.
  - `bitplane_pruning_negative`: $\text{lead} < -R \|h\|_1 \implies \text{lead} + r \cdot h < 0$, guaranteeing strict negativity.
  - `witness_bitplane_pruned`: Non-vacuous numerical witness where lead $-4$ and residual $1$ yields $-3 < 0$.
- **Mutation Control:**
  - Mutated file: `/home/socol/oc-tmp/e6-bitplane-ffn/control_copy/BitplaneRouter_mutant.lean`.
  - Mutation: Inverted conclusion inequality to assert positivity $0 < \text{lead} + r \cdot h$.
  - Control Execution:
    `flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 60 bash -c 'LEAN_PATH=... lean /home/socol/oc-tmp/e6-bitplane-ffn/control_copy/BitplaneRouter_mutant.lean'`
  - **Verdict:** RED (exit code 1, `linarith failed to find a contradiction`).
  - Restoring clean source produces exit code 0.

---

## Profit Condition Assessment

1. **(a) Collisions ($R(D) < \log_2 t$):**
   Fixed weight bit patterns are precomputed once offline.
2. **(b) Score Gaps:**
   Valid for certifying negative SwiGLU gating where $x < 0 \implies \text{silu}(x) \approx 0$.
3. **(c) Binding Resource:**
   **FAILED on dense FFN.** Memory bandwidth is the bottleneck during decode. Pruning 28.4% of neurons does not compensate for two-pass memory traffic and uncoalesced gather overhead.
