# Why gate, up and down dominate the NanoQuant error

## Result

- The three FFN linears carry **78.1%** of the pack's squared reconstruction error and **70.9%** of its bytes. Gate + up are 53.5%; down is 24.6%.
- The cause is **mass at one bit per weight**, not a bad per-matrix fit. The FFN holds 5.436B of the 8.191B served weights (66.4%) at 1.02 b/w. Its per-matrix relative error, 0.568–0.591, is the same range as q 0.705 and o 0.536. The FFN is 3.6× the weight count of attention.
- **Rank is not the lever.** A binary factor stores `r·(out + inn)` signs. At `r = out·inn/(out + inn)` it stores one sign per dense weight. Gate/up/down sit at 0.59–1.33× that point; down reaches full rank at L6 (`r = 4096`) and 3872 at L16, the rest near 3040–3300. Adding rank costs bytes linearly and cannot give the 1.7× RMS reduction that 1.5× needs.
- The **bit allocation is inverted**: embedding/head are 1,322 MB (57.3% of served bytes) at 8.5 b/w, k/v are 115 MB at 3.05 b/w for 2.1% of the error, and the FFN is 691.5 MB at 1.02 b/w for 78.1% of the error. Rebalance first, then fit the FFN through SwiGLU.
- **Target:** 1.5× is plausible with a block-wise FFN refit at fixed bits plus a modest bit reallocation. **1.15× is not reachable by FFN work alone**: with attention unchanged, zero FFN error floors the ratio at 1.20×.

All tables below are `measured` from `/home/lucas/models/arelath/analysis/layers8b-poly-0-35.csv`, except the target arithmetic, which is a first-order `model`.

## Pack allocation

| component | weights | bytes | b/w | error share |
|---|---:|---:|---:|---:|
| embedding | 622.3M | 661.2 MB | 8.50 | not in the per-linear table (Q8_0) |
| lm_head | 622.3M | 661.2 MB | 8.50 | not in the per-linear table (Q8_0) |
| attention (q,k,v,o) | 1.510B | 284.5 MB | 1.51 | 21.94% |
| FFN (gate,up,down) | 5.436B | 691.5 MB | 1.02 | 78.06% |
| **total** | **8.191B** | **2,305.7 MB** | **2.25** | **100%** |

The pack alone is 976.0 MB; the served pair adds the Q8_0 embedding and head.

## Per-section error

`energy` is the Frobenius squared error of the rebuilt served weight against the Q8_0 reference, summed over 36 layers. `rel` is the relative Frobenius error. `rank/dense` is `r` divided by `out·inn/(out + inn)`, the rank at which the factor stores one sign per dense weight. `b/w` includes the sign factors, the bf16 scales and the sparse residual.

| kind | energy | share | bytes | byte share | b/w | rel (median) | rank/dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate | 432,165.3 | 26.38% | 228.12 MB | 23.37% | 1.007 | 0.575 | 0.88–1.22× |
| up | 443,530.5 | 27.08% | 223.59 MB | 22.91% | 0.987 | 0.591 | 0.59–1.06× |
| down | 402,952.0 | 24.60% | 239.80 MB | 24.57% | 1.059 | 0.568 | 0.99–1.33× |
| q | 200,155.5 | 12.22% | 92.75 MB | 9.50% | 1.228 | 0.705 | 1.19–1.30× |
| o | 125,279.7 | 7.65% | 76.82 MB | 7.87% | 1.018 | 0.536 | 0.98–1.02× |
| k | 21,610.0 | 1.32% | 57.48 MB | 5.89% | 3.045 | 0.468 | 2.97–3.24× |
| v | 12,240.1 | 0.75% | 57.48 MB | 5.89% | 3.045 | 0.291 | 2.97–3.24× |
| **attention** | **359,285.2** | **21.94%** | **284.53 MB** | **29.15%** | **1.51** | | |
| **FFN** | **1,278,647.8** | **78.06%** | **691.51 MB** | **70.85%** | **1.02** | | |

Two readings matter:

- Gate and up have almost identical per-matrix error (0.575, 0.591). Together they are 2.17× down's energy because there are two of them at the same size and the same error. The FFN is not fitted worse than attention; q is fitted worse. The FFN is simply bigger.
- k/v are 11.8% of pack bytes for 2.1% of the error. Their rank is 2.97–3.24× the dense point, so most of their stored signs are redundant in exact arithmetic. Capping rank at the 1024 output dimension frees **66.15 MB**; capping at 768 frees 77.99 MB.

## Measured PPL

| protocol | NQ | Q4_K_M | Q8_0 | NQ / Q4 |
|---|---:|---:|---:|---:|
| ctx 64, 992 scored | 33.041986 | 14.520989 | — | 2.2755 |
| ctx 128 span 1, 1008 scored | 28.307562 | 10.988133 | 10.676001 | 2.5762 |
| ctx 128 span 2, 1008 scored | 26.318065 | 11.325463 | — | 2.3238 |
| ctx 128 combined, 2016 scored | 27.294699 | 11.155526 | — | 2.4467 |
| ctx 512, 1020 scored | 14.903005 | 6.557804 | 6.339447 | 2.2726 |
| first 31 positions | 14.641564 | 8.430632 | — | 1.7367 |

The exporter's 1.7568 is a low-context protocol; the first 31 positions here give 1.7367, consistent with it. It is not comparable to the 2.27–2.58 at scoring contexts 64–512. The bar is **~2.3×**, not 1.76×.

## Mechanism

1. **Mass.** The FFN is 5.436B of 8.191B served weights (66.4%) and 70.9% of pack bytes. At a similar relative error per matrix, it produces 78.1% of the squared error. Gate + up are two 12288×4096 matrices per layer (100.7M weights); down is one 4096×12288 matrix (50.3M). That 2:1 ratio is the 53.5:24.6 split.

2. **Bit depth, not rank.** The served weight is `s_post · U · V · s_pre` with `U, V ∈ {±1}` plus five (gate/up) or thirteen (down) sparse corrections per output row. At `r ≈ out·inn/(out+inn)`, the factor stores one sign per dense weight; that is ~1 b/w. Gate/up/down are at 0.59–1.33× that point. The relative error ~0.57 is the error of a one-bit sign factorization, not of a rank truncation. The sparse residual is a polish: at L35 gate, `lowrel` 0.575139 vs `rel` 0.574624, a 0.09% reduction. More rank at one bit per sign buys little; more bits or a better fit buys a lot.

3. **SwiGLU coupling.** With `y = down(silu(gate(h)) ⊙ up(h))`, a first-order perturbation is

   `δy = down( silu'(g) ⊙ δg ⊙ u + silu(g) ⊙ δu ) + δdown( silu(g) ⊙ u )`.

   The gate and up errors enter the same 12288-dimensional product, so their variances add before the nonlinearity. Down's error is added after it. The branch gains are O(1) and depend on the gate preactivation; the block fit measures them directly. Gate + up are therefore both the largest mass and the earliest point in the FFN.

4. **Depth.** The FFN runs in all 36 blocks. Its share of each layer's error is 71.6% at L0, 62.0% at L1, 52.9% at L2, then 76–82% for L4–L35 (77.3% at L35). The FFN is 16.7–23.2 MB of the 24.5–31.1 MB pack per layer. Later layers are slightly worse fits, not larger ones.

The 78.1% is weight-space error. The end-to-end NLL depends on the activation-weighted output error, which is the quantity the block fit optimizes; the weight-space share is a proxy, not the objective.

## Target arithmetic

Working midpoint 2.30 (measured 2.27–2.58), excess `ln 2.30 = 0.833`. FFN share 0.7806, attention 0.2194. If FFN squared error is multiplied by `r` and attention by `a`, the first-order model is

`ratio ≈ exp(0.833 · (0.7806·r + 0.2194·a))`.

| FFN squared error | attention unchanged | attention ×0.5 | attention ×0.25 |
|---|---:|---:|---:|
| ×1.00 | 2.30 | 2.10 | 2.01 |
| ×0.50 | 1.66 | 1.52 | 1.45 |
| ×0.343 | **1.50** | 1.37 | 1.31 |
| ×0.25 | 1.41 | 1.29 | 1.23 |
| ×0.15 | 1.32 | 1.21 | **1.15** |
| ×0.00 | 1.20 | 1.10 | 1.05 |

Consequences:

- **1.5×** needs FFN squared error at 34% of today's (RMS ×0.585) with attention unchanged. Across the measured 2.27–2.58 base the factor is 0.30–0.37.
- **1.3×** needs FFN at 12% (RMS ×0.350).
- **1.15×** needs FFN at ~15% *and* attention at ~25%. The FFN alone cannot get there: zero FFN error floors the ratio at 1.20×.

The model is linear in squared error and transfers the weight-space shares to output space. It is a planning bound, not a measurement.

## What to do, in order

1. **One-layer bake-off before renting a box.** Fit L0 and L35 `gate`, `up` and `down` at 1.0, 1.33 and 2.0 b/w with the existing `factorize` (`src/learn/nanoquant.rs`). Measure weight-space and activation-weighted output error. This answers whether extra rank helps or the binary format saturates. One 50M matrix per run, minutes to hours on CPU; no 8B run.

2. **Block-wise FFN refit through SwiGLU.** The repo already has the right tool: `BlockTune` in `src/gpu/nn/train/block.rs`, one residual block's o-projection and SwiGLU FFN, Adam through the straight-through sign, loss on the teacher's block output with per-output importance (`compress_block.py`). Run `gate` and `up` first (53.5% of the error, shared 4096-dim input), then `down` on the fitted intermediate distribution. This is the activation-aware refit; it optimizes the function, not the weights, at fixed bits. It is Vulkan-only; the reference loop is ported and exercised on 0.6B in `tests/learn_block_tune.rs`, and the 8B run is the missing experiment. A CPU fallback is a per-linear weighted fit with the input covariance, then a down correction.

3. **Down correction.** After the gate/up refit, fit `down` to the *fitted* intermediate `ẑ = silu(ĝ)⊙û` and the teacher FFN output, not to the teacher's down weights. The least-squares target is `D̂ = E[y ẑᵀ] E[ẑ ẑᵀ]⁻¹`, projected back onto the NQ format. This absorbs the part of the gate/up error correlated with `ẑ` at zero extra bytes. It targets the 24.6% down share and is much cheaper than a full block tune.

4. **Rebalance the bits.** Embedding/head at 8.5 b/w are 1,322 MB. A 6-bit payload (~6.5 b/w) saves ~311 MB; a 4-bit payload (~4.5 b/w) saves ~622 MB and, if the freed bytes are not reallocated, takes the served pair to ~1.68 GB, meeting the 2 GB target. Q4_K_M quantizes the same tensors and its total PPL is 1.03× Q8_0, so the PPL cost is small. k/v capped at 1024 frees another 66 MB. Spend the freed bytes on FFN up to full rank and on q/o. For the 12288×4096 and 4096×12288 shapes, full rank is 1.33 b/w; above that the extra rank is an overcomplete binary dictionary with diminishing returns. q and o are 4096×4096, where full rank is 2.0 b/w, so they can absorb more. Do not assume freed bytes convert one-for-one into FFN quality; hold the rest for a second bit plane or a larger residual.

5. **QAT/KD only if 1–4 saturate above 1.5.** The block fit is already a small QAT. The remaining gap to 1.15 is the binary format; a second bit plane, block scales, or full-model distillation is the later, GPU-heavy option.

## What not to do

- Do not spend the next run on FFN rank at the current 1 b/w. Gate/up/down are at 0.59–1.33× the dense point; a 66 MB rank bump is +0.8% rank. It cannot produce the 1.7× RMS reduction that 1.5× needs.
- Do not use the exporter's 1.7568 as the bar. At scoring contexts 64–512 the same artifact measures 2.27–2.58.
- Do not optimize weight-space Frobenius error alone. The PPL depends on the activation-weighted output error; the block fit is the right metric.

## Not covered

- The full wikitext-2 test set (~245k tokens) is not measured; the PPL values are 2048-token spans. The ratio is the comparison, not the absolute PPL.
- No refit has been run. The 1.5/1.15 projections are the first-order model above.
- The Q8_0 reference has its own quantization noise; `rel` is measured against it, not against the original f16 weights.
- `BlockTune` needs a Vulkan GPU and an 8B teacher dump; neither is present on this machine. The local GPU tests fail for missing `glslc`/Vulkan ICD.
