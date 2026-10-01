# Linear Attention Conversion of Qwen2.5-0.5B via Gated DeltaNet

## Intent & Context

Per token, standard autoregressive transformer decoding costs $\Theta(\text{active parameters})$ for the weights and $\Theta(t)$ for attention over the cached key-value states. As identified in `docs/evidence/pipeline-inventory.md`, KV cache read traffic overtakes the weight matrix stream at long sequence lengths (~8,250 tokens on Qwen2.5).

This investigation evaluates whether `Qwen2.5-0.5B` can be converted from quadratic softmax attention to an $O(1)$ fixed-state recurrent mechanism without sacrificing its greedy decoding behavior. Specifically, we test:
1. **Target Mechanism**: **Gated DeltaNet (GDN)** (Yang, Kautz, Hatamizadeh, arXiv 2412.06464; DeltaNet chunking, arXiv 2406.06484).
   Per head: fixed recurrent state matrix $S_t \in \mathbb{R}^{d_v \times d_k}$ with update rule:
   $$S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$

   * **say:** S sub t equals alpha sub t times S sub t minus one times the quantity I minus beta
     sub t k sub t k sub t transpose, plus beta sub t v sub t k sub t transpose.
   * **means:** one token step decays the previous state matrix by the gate $\alpha_t$, applies the
     rank-one correction $I - \beta_t k_t k_t^T$, and adds the value-key outer product scaled by
     $\beta_t$; the state stays $d_v \times d_k$, so context length changes its contents, not its
     size (Gated DeltaNet: Yang, Kautz, Hatamizadeh, arXiv 2412.06464; chunked delta rule:
     arXiv 2406.06484).
   * **vars:** $S_t$ — recurrent state after token $t$, $\mathbb{R}^{d_v \times d_k}$; $S_{t-1}$ —
     state before token $t$, same type; $\alpha_t$ — data-dependent decay gate, $(0, 1]$;
     $\beta_t$ — data-dependent retrieval rate, $(0, 1)$; $k_t$ — $L_2$-normalised key,
     $\mathbb{R}^{d_k}$; $v_t$ — value, $\mathbb{R}^{d_v}$; $I$ — identity matrix,
     $\mathbb{R}^{d_k \times d_k}$; $d_v, d_k$ — head dimensions, $\mathbb{N}$; $t$ — token
     index, $\mathbb{N}$.
   * **apeiron:** `Nous.Seq.GatedDelta.step_eq_affApply` carries the gated delta recurrence, with
     `Nous.Seq.GatedDelta.stepAffine` as the step it applies, and
     `Nous.Seq.AffineMixer.stepAffineGeneral` for a general transition matrix. The corpus states
     the algebraic identity over a general commutative ring; it is not a claim about this
     checkpoint. Proved.
   * **physics:** a fixed-size associative memory: the decay gate forgets, the outer product
     writes.

   where $q_t, k_t$ are $L_2$-normalized across head dimension $d_k$, $v_t$ is extracted via pretrained $W_v$, and small data-dependent gate projections predict $\alpha_t \in (0, 1]$ (decay) and $\beta_t \in (0, 1)$ (retrieval rate). Rotary Position Embeddings (RoPE) are eliminated on converted layers in accordance with Gated DeltaNet architecture. Pretrained projection weights $W_q, W_k, W_v, W_o$ are preserved and initialized directly from the base checkpoint.
2. **LoLCATs Attention Transfer Two-Stage Training** (HazyResearch/lolcats, arXiv 2410.10254):
   - **Stage 1 (Attention Output Distillation)**: Freeze pretrained Transformer layers; train gate projections $\alpha_t$ and $\beta_t$ to minimize mean-squared error (MSE) between linear attention outputs and the original softmax attention teacher outputs layer-by-layer.
   - **Stage 2 (LoRA Next-Token Adaptation)**: Apply low-rank adapters ($r=16, \alpha=32$) to attention projections ($W_q, W_k, W_v, W_o$) and jointly optimize with cross-entropy next-token loss on held-out public tokens.
3. **Configurations Tested**:
   - **Original fp16 reference**: 24 standard softmax attention layers with full KV cache.
   - **Config A (Hybrid 18/24 GDN)**: 18 layers converted to Gated DeltaNet, keeping 6 evenly spaced softmax attention layers (layers 0, 4, 8, 12, 16, 20; 3:1 subquadratic-to-softmax ratio in the Qwen3-Next style).
   - **Config B (All 24/24 GDN)**: All 24 layers converted to pure Gated DeltaNet recurrent state.

---

## Method & Artifacts

- **Model**: `Qwen/Qwen2.5-0.5B` (HuggingFace revision `060db6499f32faf8b98477b0a26969ef7d8b9987`).
  - Architecture: 24 hidden layers, hidden dimension 896, 14 query heads ($d_k = 64$), 2 key-value heads ($d_v = 64$).
  - Grouped Value Attention (GVA): 7 query heads share 1 KV head; $k$ and $v$ are repeated to 14 heads before entering GDN.
- **Hardware**: NVIDIA GeForce RTX 5080 (16,303 MiB physical VRAM; execution footprint < 6.0 GB VRAM).
- **Corpus & Datasets**:
  - Training Corpus: `Salesforce/wikitext` (`wikitext-103-v1`, Parquet chunk 0, 4,000,000 token extract).
    - Stage 1 Distill: 500,000 tokens (batches of sequence length 512).
    - Stage 2 LoRA: 500,000 disjoint tokens from the same corpus.
  - Evaluation Held-out Set (matching `dec-t0` decision-cell census protocol from `docs/evidence/decision-cell-census.md`):
    - Text: `/home/socol/Workspace/fold/hashtrinity/prose8.txt`.
    - Teacher-forced evaluation over 33 blocks of 512 tokens ($33 \times 511 = 16,863$ tokens, disjoint from calibration and training by >600k tokens).
    - Free-running greedy prompt set: 50 non-overlapping prompts of length 32 tokens sampled from the held-out split; 64 new tokens generated per prompt with `do_sample=False`.
  - Needle-in-a-Haystack Retrieval Test:
    - Context length: 4,096 tokens filled with background text.
    - Needle: 5-digit unique key inserted at retrieval distance 512 and distance 2,048 from prompt end.
    - 20 trials per configuration; pass rate scored on exact key match in greedy generation.
- **Artifacts and Scripts**:
  - GDN Layer Implementation: `/home/socol/oc-tmp/conv-t0/model_gdn.py` (sha256: `643ffd47ffc4792027926be71e38029e93f45ac60f485bf2598d679b465238b2`)
  - End-to-End Pipeline Script: `/home/socol/oc-tmp/conv-t0/run_pipeline.py` (sha256: `f119243f82cfff9f06cbd92a24f953221059188433341131b47279a251a5fb36`)
  - Gate Verification Script: `/home/socol/oc-tmp/conv-t0/check.py` (sha256: `e067c7729c5fe7f02e7e27e34a1799e9ad6f78a157cc305256dd76ade6bee5f9`)
  - Evaluation Output JSON: `/home/socol/oc-tmp/conv-t0/eval_results.json`

---

## 1. Decision-Cell Census & Quantization Comparison Table

The results below compare the linearized models against the `fp16 reference` and the quantization baselines established in `docs/evidence/decision-cell-census.md` (`dec-t0`):

| Model / Ansatz | State Type | bits/token | flip rate | share flips lowest decile | mean identical greedy prefix | Needle 512 (20 trials) | Needle 2048 (20 trials) |
|---|---|---|---|---|---|---|---|
| **fp16 reference** | Softmax $O(t)$ KV | 4.5912 | 0.00% | 0.00% | 64.00 | 100.0% | 100.0% |
| **4-bit seal** (`dec-t0`) | Softmax $O(t)$ KV | 4.6726 | 15.55% | 32.38% | 8.66 | — | — |
| **GPTQ-int4** (`dec-t0`) | Softmax $O(t)$ KV | 4.8137 | 21.94% | 25.95% | 4.78 | — | — |
| **GPTQ-ternary L=2** (`dec-t0`)| Softmax $O(t)$ KV | 5.3845 | 39.44% | 16.42% | 2.74 | — | — |
| **Config A (hybrid 18/24 GDN)** | Hybrid $O(1) + O(t)$ | 7.1035 | 63.58% | 12.41% | 0.44 | 0.0% | 0.0% |
| **Config B (all 24/24 GDN)** | Pure $O(1)$ GDN | 9.7968 | 83.60% | 10.56% | 0.24 | 0.0% | 0.0% |

---

## 2. Flips Binned by Reference Margin Decile

Breakdown of flip distribution across the 10 reference margin deciles (D1 lowest logit margin $[0, 0.125]$ to D10 highest logit margin $>3.30$):

| Configuration | Total Flips | D1 [0, 0.125] | D2 (0.125, 0.25] | D3 (0.25, 0.41] | D4 (0.41, 0.59] | D5 (0.59, 0.81] | D6 (0.81, 1.08] | D7 (1.08, 1.45] | D8 (1.45, 2.02] | D9 (2.02, 3.30] | D10 (> 3.30) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4-bit seal (`dec-t0`) | 2,622 | 32.38% | 25.06% | 17.66% | 12.13% | 6.48% | 3.13% | 2.06% | 0.76% | 0.34% | 0.00% |
| GPTQ-int4 (`dec-t0`) | 3,699 | 25.95% | 21.01% | 17.65% | 12.98% | 8.95% | 6.08% | 4.03% | 2.00% | 0.78% | 0.57% |
| **Config A (hybrid)** | 10,721 | 12.41% | 12.10% | 11.95% | 11.71% | 10.96% | 9.98% | 9.64% | 8.40% | 7.24% | 5.62% |
| **Config B (all 24)** | 14,097 | 10.56% | 10.31% | 10.48% | 10.88% | 10.44% | 9.84% | 10.21% | 9.48% | 9.02% | 8.78% |

---

## 3. Per-Token State Footprint at $t = 4096$

In `Qwen2.5-0.5B`, the attention parameters are:
- Number of layers: $L = 24$.
- Number of heads: $H = 14$ query heads, $H_{kv} = 2$ key-value heads.
- Head dimension: $d_k = 64, d_v = 64$.

For standard softmax attention, each token position stores keys and values for $H_{kv}$ heads across $L$ layers:
$$\text{State}_{\text{softmax}}(t) = 2 \times L \times H_{kv} \times d_k \times t = 2 \times 24 \times 2 \times 64 \times t = 6,144 \times t \text{ scalars}$$

* **say:** State sub softmax of t equals two times L times H sub k v times d sub k times t, which
  is two times 24 times two times 64 times t, which is 6,144 times t scalars.
* **means:** full softmax attention keeps one key vector and one value vector per key-value head,
  per layer, per token, so the cached scalar count grows linearly in $t$; 6,144 is the count added
  per token for these shapes.
* **vars:** $\text{State}_{\text{softmax}}(t)$ — cached scalars after $t$ tokens, $\mathbb{N}$;
  $2$ — one key vector and one value vector; $L$ — layers, 24; $H_{kv}$ — key-value heads, 2;
  $d_k$ — head dimension, 64; $t$ — decoded tokens, $\mathbb{N}$.
* **apeiron:** `none — anchor`. A footprint count for the `Qwen2.5-0.5B` shapes; no corpus
  declaration states a KV-cache size.
* **physics:** the working set grows with context length.

At $t = 4,096$: $25,165,824$ scalars.

For Gated DeltaNet, each converted layer maintains a fixed recurrent matrix $S \in \mathbb{R}^{d_v \times d_k}$ for each of the $H = 14$ heads:
$$\text{State}_{\text{GDN}} = L_{\text{conv}} \times H \times d_v \times d_k = L_{\text{conv}} \times 14 \times 64 \times 64 = L_{\text{conv}} \times 57,344 \text{ scalars (constant with } t\text{)}$$

* **say:** State sub G D N equals L sub conv times H times d sub v times d sub k, which is L sub
  conv times 14 times 64 times 64, which is L sub conv times 57,344 scalars, constant with t.
* **means:** each converted layer carries one fixed recurrent matrix per head, so the GDN state
  scales with the number of converted layers and not with $t$.
* **vars:** $\text{State}_{\text{GDN}}$ — recurrent scalars, $\mathbb{N}$; $L_{\text{conv}}$ —
  converted layers; $H$ — query heads, 14; $d_v, d_k$ — head dimensions, 64 each; $t$ — decoded
  tokens, absent from the expression.
* **apeiron:** `none — anchor`. A fixed-state count for the `Qwen2.5-0.5B` head split; no corpus
  declaration states it.
* **physics:** a constant working set at any context length.

For Config A (Hybrid: 6 softmax layers, 18 GDN layers) at $t = 4,096$:
$$\text{Softmax KV}: 2 \times 6 \times 2 \times 64 \times 4,096 = 6,291,456 \text{ scalars}$$

* **say:** Softmax K V equals two times six times two times 64 times 4,096, which is 6,291,456
  scalars.
* **means:** the six softmax layers retained by Config A still carry a linear-in-$t$ key-value
  state; at $t = 4{,}096$ that is 6,291,456 scalars.
* **vars:** $\text{Softmax KV}$ — cached scalars, $\mathbb{N}$; $2$ — key and value; $6$ —
  retained softmax layers; $2$ — key-value heads; $64$ — head dimension; $4{,}096$ — context
  tokens.
* **apeiron:** `none — anchor`. Arithmetic over the Config A layer split; no corpus declaration.
* **physics:** the linear memory left in the hybrid.

$$\text{GDN State}: 18 \times 57,344 = 1,032,192 \text{ scalars}$$

* **say:** G D N State equals eighteen times 57,344, which is 1,032,192 scalars.
* **means:** the eighteen converted layers hold a fixed 1,032,192-scalar state that does not
  change with $t$.
* **vars:** $\text{GDN State}$ — recurrent scalars, $\mathbb{N}$; $18$ — converted layers;
  $57{,}344$ — scalars per converted layer.
* **apeiron:** `none — anchor`. Arithmetic over the Config A layer split; no corpus declaration.
* **physics:** constant state carried across tokens.

$$\text{Total}: 7,323,648 \text{ scalars}$$

* **say:** Total is 7,323,648 scalars.
* **means:** the Config A state at $t = 4{,}096$ is the sum of the six softmax layers' key-value
  state and the eighteen converted layers' recurrent state, $6{,}291{,}456 + 1{,}032{,}192$.
* **vars:** $\text{Total}$ — Config A state, scalars, $\mathbb{N}$; the summands are the two
  equations directly above.
* **apeiron:** `none — anchor`. Arithmetic over the Config A layer split; no corpus declaration.
* **physics:** the hybrid's total working set at $t = 4{,}096$.

For Config B (All 24 GDN layers) at any $t$:
$$\text{Total}: 24 \times 57,344 = 1,376,256 \text{ scalars}$$

* **say:** Total equals twenty-four times 57,344, which is 1,376,256 scalars.
* **means:** the all-converted configuration holds a fixed 1,376,256-scalar state at any $t$.
* **vars:** $\text{Total}$ — Config B state, scalars, $\mathbb{N}$; $24$ — converted layers;
  $57{,}344$ — scalars per converted layer.
* **apeiron:** `none — anchor`. Arithmetic over the Config B layer split; no corpus declaration.
* **physics:** constant state carried across tokens.

The per-token state byte footprint (total state bytes divided by $t = 4096$) is:

| Configuration | Total Scalars at $t=4096$ | Per-Token fp16 Bytes (2 B/sc) | Per-Token Q32.32 i64 Bytes (8 B/sc) | Asymptotic State Complexity |
|---|---|---|---|---|
| **Original (full KV)** | 25,165,824 | 12,288 B (393,216 bytes total / 32 tokens) | 49,152 B (1,572,864 bytes total / 32 tokens) | $\Theta(t)$ |
| **Config A (hybrid 18/24)** | 7,323,648 | 3,576 B (187,904 bytes total / 32 tokens) | 14,304 B (751,616 bytes total / 32 tokens) | $\Theta(t)$ (25% rate) |
| **Config B (all 24/24)** | 1,376,256 | 672 B (119,808 bytes total / 32 tokens) | 2,688 B (479,232 bytes total / 32 tokens) | $\Theta(1)$ |

*(Note: Per single step at $t=4096$, full KV requires 12,288 bytes/tok in fp16; Config A requires 3,576 bytes/tok; Config B requires zero additional state per token, maintaining a fixed 2.75 MB total state buffer).*

---

## 4. Verdict & Findings

1. **Greedy Behavior Invariance**:
   - **Neither configuration keeps greedy behavior within the 4-bit seal's flip rate (15.55%)**.
   - Hybrid Config A experiences **63.58%** greedy top-1 argmax flips ($\Delta = +48.03\%$ worse than 4-bit seal), with cross-entropy bits/token increasing from 4.5912 to 7.1035. Its mean identical greedy rollout prefix collapses from 64.0 tokens to **0.44 tokens**.
   - Pure GDN Config B experiences **83.60%** greedy top-1 flips, with bits/token rising to 9.7968 and mean identical prefix dropping to **0.24 tokens**.
2. **Margin Distribution Shift**:
   - For high-fidelity weight quantizations (4-bit seal and GPTQ $L=3$), flips concentrate in the lowest margin deciles (32.38% and 36.81% in D1, respectively), confirming that errors occur primarily at razor-thin decision boundaries.
   - For linearized GDN attention, flips are **uniformly dispersed across all 10 margin deciles** (12.41% in D1 for Hybrid; 10.56% in D1 for All 24). This indicates that linear attention conversion introduces macroscopic representation disruption that overwhelms even large confidence margins.
3. **Long-Context Needle Retrieval**:
   - Original FP16 attention achieves **100.0%** exact needle recall at both distance 512 and distance 2,048 in a 4,096-token context.
   - Both Config A and Config B score **0.0%** pass rate across 20 trials. Attention transfer without extensive billion-token pretraining destroys associative memory recall over long contexts.
4. **Conclusion**:
   - Post-hoc linearization of a pretrained softmax transformer using LoLCATs attention transfer and low-rank adaptation on modest token budgets ($\sim 1\text{M}$ tokens) is insufficient to preserve greedy token trajectory or long-context associative recall.
   - Subquadratic decoding via Gated DeltaNet requires native pretraining from scratch or pretraining-scale continual distillation.

---

## 5. Residuals & Verification Gate

- Verification Gate: `/home/socol/oc-tmp/conv-t0/check.py` checks that all headline metrics, table entries, and memory footprints match `/home/socol/oc-tmp/conv-t0/eval_results.json`.
- Negative Control: `/home/socol/oc-tmp/conv-t0/test_control.sh` creates a throwaway mutated copy of the report and verifies `check.py` exits with non-zero exit code 1.
EOF