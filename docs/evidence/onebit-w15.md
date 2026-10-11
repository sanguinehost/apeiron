# ONEBIT W15a: direct KL along the quantisation ray

Command (from the repository root, `/home/socol/Workspace/spinor-ttl/.venv/bin/python`, RTX 5080):
`python tools/onebit/ray_scan.py /home/socol/oc-tmp/escalation/2026-10-11-stiffness.json /home/socol/oc-tmp/escalation/2026-10-11-ray.json 100`.
100 C4 chunks of 512 tokens (`/home/socol/oc-tmp/seal/calib-c4.txt`), wall 139.6 s. The 21 `ffn_down` tensors of
Qwen2.5-0.5B move jointly to `W + tD` (`D` the arm's dequantised error, `fisher_capture.load_deltas`), and the
per-token KL `KL(p_W ‖ p_{W+tD})` is computed in float64.

- Source: `tools/onebit/ray_scan.py` sha256 `ad65df8f0cc58c2d…`; receipt sha256 `40e693d217fdbe6c…`.
- Prediction `ray-1` was withdrawn, not scored: the run was launched as `cd tools/onebit && python ray_scan.py`, so the
  derivation gate's run class did not admit it. The numbers below are `measured`; they carry no prediction verdict.

## Result

`K` is the W14 sampled-Fisher joint second-order KL per token; `r(t) = KL(t) / (t² K)`.

| arm | `KL(1/8)` | `KL(1)` | `KL(−1)` | `64·KL(1/8)` (exact quadratic) | W14 `K` | `KL(1)/(64·KL(1/8))` |
|---|---:|---:|---:|---:|---:|---:|
| `IQ1_S` | 0.0061071 | 0.449916 | 0.501859 | 0.3909 ± 0.0064 | 0.4573 | 1.151 |
| `IQ1_M` | 0.0047723 | 0.344035 | 0.371531 | 0.3054 ± 0.0051 | 0.3299 | 1.126 |
| Q4_K_M | 0.0000704 | 0.004501 | 0.004542 | 0.00451 ± 0.00007 | 0.003965 | 0.998 |
| RTN | 0.0555673 | 9.82565 | 1.59422 | 3.556 ± 0.058 | 3.553 | 2.763 |
| `had-gptq1` | 0.0065456 | 0.460010 | 1.069030 | 0.4189 ± 0.0084 | 0.2919 | 1.098 |

(± is the standard error over the 100 chunks, scaled by 64.)

- **The quadratic form holds to the full step at one-bit error.** `KL(1) / (64·KL(1/8))` is 0.998–1.151 for every
  arm except RTN (2.763). Higher-order terms add at most 15% at `t = 1` on `IQ1_S`, `IQ1_M` and `had-gptq1`.
- **The W14 rank inversion was estimator error.** The exact quadratic orders `had-gptq1` (0.419) > `IQ1_S` (0.391) >
  `IQ1_M` (0.305), the P1 KLD order. The W14 400-sample estimator read `had-gptq1` low by 1.43× and `IQ1_S` high by
  1.17× (`r(1/8)` 1.435 and 0.855).
- **Odd remainder.** At `t = 1`, `odd = (KL(D) − KL(−D))/2` is −0.305 on `had-gptq1` (its error lies on the downhill
  side of a cubic term) and +4.12 on RTN.

## Formal record

`Perf.CostCorrespondence.ray_w15a` (`proof/corpus/Perf/CostCorrespondence.lean`, `proved`, closed cone) states the
three verdicts on these figures as ℚ facts; a changed `had-gptq1` bound (`< 1.09`) reddens it
(`/home/socol/oc-tmp/costcorr-wf-controls.log`).

## What follows

- The water-filling floor (`waterfill_floor`) applies to one-bit error levels on `ffn_down`: the quadratic form is the
  objective to within 15%.
- The sampled-Fisher estimator is too noisy to rank arms. The Kronecker factorisation `G ⊗ H` (`kron_diag`) is the
  next instrument, validated against `64·KL(1/8)` above before any floor is read from it.

## Not covered

- `ffn_down` only, one model (Qwen2.5-0.5B), C4 calibration text; P1 KLD is wikitext-2 through llama.cpp.
- `t ∈ {±1/8, 1/4, ±1/2, ±1}`; the remainder between those points is not bounded.

# ONEBIT W15b: K-FAC water level over every linear tensor

Command (repository root, same venv and GPU): `python tools/onebit/kfac_floor.py
/home/socol/oc-tmp/escalation/2026-10-11-kfac.json 100 4`. 100 C4 chunks, 4 sampled targets per token, all 168 linear
tensors (7 kinds × 24 blocks). Wall 118.3 s (capture 22.0 + 17.8 s in two 12-block halves, analysis 45.0 s).

- Source: `tools/onebit/kfac_floor.py` sha256 `1fb742d7080c0417…`; receipt sha256 `e64115b13698bf50…`; log
  `/home/socol/oc-tmp/escalation/2026-10-11-kfac.log`.
- Control: the gguf f16 weights equal the HF weights the factors are captured on, max relative difference 5.0e-8.
- Prediction `kfac-1-1`: **refuted** on its order clause; its factor, rate, oracle-below-separable and control
  clauses held.

## Method

- Per tensor, `G` is the per-token covariance of the output gradient of `log p(y)` with `y` sampled from the model,
  through the full backward graph (the embedding output is the only leaf). `H` is the input covariance, shared per
  input (`attn_in`, `o_in`, `mlp_in`, `down_in`).
- Per-token KL of an error `D` is read as `½ tr(DᵀGDH)` (`model`: drops cross-token covariance and the dependence of
  `g` on `x`).
- Floors: in the eigenbases `G = U diag(g) Uᵀ`, `H = V diag(h) Vᵀ`, the coefficients `C = UᵀWV` carry weights
  `½ g_i h_j` (`kron_diag`), and global reverse water-filling (`waterfill_opt`, `waterfill_level`) over all 168
  tensors gives `KL(R) = Σ min(λ, θ)` at rate `R = Σ max(0, ½ log₂(λ/θ))`.
  - **separable**: `σ²_ij = r_i c_j / m`, the variance a row-and-column-scaled Gaussian codebook achieves (Lapidoth
    1997, `anchor`); scale side information 16 bits per row and column, counted.
  - **oracle**: `σ²_ij = C_ij²`, an optimistic reference, not a bound.
  - The basis `(U, V)` is not counted in the rate.

## Validation against the exact quadratic

Sum over the 21 `ffn_down` tensors (blocks 3–23) against `64·KL(1/8)` from W15a:

| arm | K-FAC | exact quadratic | ratio |
|---|---:|---:|---:|
| `IQ1_S` | 0.4233 | 0.3909 | 1.083 |
| `IQ1_M` | 0.3366 | 0.3054 | 1.102 |
| Q4_K_M | 0.003621 | 0.004509 | 0.803 |
| RTN | 10.40 | 3.556 | 2.924 |
| `had-gptq1` | 0.2764 | 0.4189 | 0.660 |

- K-FAC reads the four structured arms within 0.66–1.10 of the exact quadratic. RTN is out of the quadratic regime
  (W15a `KL(1)/(64·KL(1/8))` 2.763).
- K-FAC orders `had-gptq1` lowest where the exact quadratic orders it highest. GPTQ shapes its error against `H`
  alone, which is the factor the Kronecker model treats as independent of `g`; the under-read is that assumption,
  not sampling noise (`model`).

## Floor

Q4_K_M's K-FAC KL over all 168 tensors is 0.015675 (q 0.00042, k 0.00074, v 0.00297, o 0.00131, gate 0.00198,
up 0.00398, down 0.00427).

| rate (b/w) | separable KL | oracle KL |
|---:|---:|---:|
| 1.0 | 0.3470 | 0.1437 |
| 1.18 | 0.2645 | 0.1038 |
| 1.5 | 0.1658 | 0.0601 |
| 2.0 | 0.0814 | 0.0269 |
| 2.5 | 0.0403 | 0.0126 |
| 3.0 | 0.0200 | 0.00598 |
| 4.0 | 0.00497 | 0.00141 |
| 4.5 | 0.00248 | 0.000695 |

- Rate that matches Q4_K_M's KL: **3.176 b/w separable, 2.353 b/w oracle**.
- At 1.18 b/w the floor is 16.9× (separable) and 6.6× (oracle) Q4_K_M's KL.

## Defect found: truncated backward in `fisher_capture.py`

`tools/onebit/fisher_capture.py:79` replaces every captured `down_proj` output with a detached leaf. The gradient at
block `b` then flows through no later block's MLP branch, so the W14 per-block scores used a truncated gradient.
`kfac_floor.py` detaches only the embedding output. The full-graph K-FAC still under-reads `had-gptq1` (0.660), so the
truncation is not the whole of the W14 inversion.

## Formal record

`Perf.CostCorrespondence.kfac_w15b` (`proved`, closed cone) states the validation band, the refuted order, the
16×/6× floor factors and the matching rates as ℚ facts. Controls: the order flipped and the separable factor raised to
17× each redden it (`/home/socol/oc-tmp/costcorr-kfac-controls.log`).

## What follows

- On Qwen2.5-0.5B, post-training coding in the K-FAC metric needs ≥ 2.35 b/w (oracle) to reach Q4_K_M's KL; 1.18 b/w
  is 6.6× above Q4 even there (`model`).
- The open question for the 27B target is how the matching rate scales with model size. The next measurement is the
  same floor on larger Qwen2.5 checkpoints, registered before it runs.

## Not covered (W15b)

- One model, 0.5B; embeddings and `lm_head` excluded; cross-token terms and `g`–`x` dependence dropped.
- Gaussian-codebook rate model; basis side information uncounted; the oracle is not achievable.
- C4 calibration; P1 KLD is wikitext-2 through llama.cpp.

# ONEBIT W15c: the converse side and the floor against model size

Command (repository root, same venv and GPU): `python tools/onebit/converse.py {controls,main,ekfac,scale} …`.
Source `tools/onebit/converse.py` sha256 `29034f02d9002f6f…`; receipts
`/home/socol/oc-tmp/escalation/2026-10-11-converse-{controls,main,ekfac,scale-*}.json` with `.log`.

## Measured structure a 1.18 b/w code could exploit (Qwen2.5-0.5B, 168 tensors)

`slb_gap`: under the Shannon lower bound (`anchor`, Shannon 1959), a code that reaches Q4_K_M's KL at 1.18 b/w needs
negentropy `J ≥ R_g(D_Q4) − 1.18`: 1.994 b/w on the separable curve, 1.173 b/w on the oracle curve.

| structure | measured | estimator control |
|---|---:|---|
| marginal negentropy, K-FAC coefficient basis at the Q4 level | 6.9e-5 b/component | Gaussian 3.6e-5; Laplace 0.10465 (truth 0.10440) |
| marginal negentropy, raw weight basis | 0.00815 b/component | as above |
| causal-neighbour context gain | −0.031 b | iid-Laplace −0.032 b (estimator noise) |
| linear structure (Gaussian total correlation over the shuffled null) | 0.141 b/w (`attn_q` 0.77, `attn_o` 0.53, `attn_k` 0.38, FFN 0.05–0.10) | AR(1) −0.4%, iid −0.3% |

- Measured `J` < 0.15 b/w against the 1.994 (separable) and 1.173 (oracle) b/w required. On 0.5B, marginal, context
  and linear structure do not close the gap. Nonlinear structure was not probed.
- `K0` from log-determinants is 1.1918, within 15% of the bisection `K0` (1.227 at 4.5 b/w, 1.312 at 1.18 b/w).
- Eigenvalue-corrected K-FAC moves `had-gptq1` from 0.659 to 0.701 of the exact quadratic; the order inversion
  persists (prediction `conv-ekfac-1` refuted).

## The Q4-matching rate falls with model size

Same pipeline; Q4_K_M made locally for 1.5B and 3B (llama.cpp imatrix on 100 C4 chunks at c = 512, then quantize;
files in `/home/socol/oc-tmp/converse/`). bf16 against fp32 on 0.5B changes `R*` by 0.0016.

| model | layers | Q4_K_M K-FAC KL (full-model est.) | `R*` high-res | `R*` histogram bisection |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | 24 of 24 | 0.015697 | 3.147 | **3.174** |
| Qwen2.5-0.5B | 8 of 24 (0, 3, 6, 10, 13, 16, 20, 23) | 0.019461 | 3.003 | 3.051 |
| Qwen2.5-1.5B | 8 of 28 (0, 4, …, 24, 27) | 0.017634 | 2.724 | **2.780** |
| Qwen2.5-3B | 8 of 36 (0, 5, …, 35) | 0.018151 | 1.670 | **2.378** |

- `R*` is the all-in rate (side information counted) at which separable water-filling reaches Q4_K_M's KL.
- Subset control (prediction `subset-bias-1`, refuted on its KL clause only): 8 stratified layers read `R*` 0.12 b/w
  low and the Q4 KL 24% high on 0.5B. Corrected by that offset, the trend is 3.17 → ≈2.90 → ≈2.50 b/w.
- At 3B the high-resolution formula drops near-null components the water level does not reach; the bisection column
  is the measurement.
- `K0/N` falls 3.33e-9 → 5.76e-10 → 6.53e-11 while Q4_K_M's KL stays at 0.016–0.019: Q4_K_M spends a nearly
  constant KL, and the curvature per weight it spends it against falls as the model grows.

## Formal record

`Perf.CostCorrespondence.{waterfill_highres, waterfill_exclusion, slb_gap, converse_w15c}` (`proved`, closed cone,
`#audit`ed in `proof/corpus/AxiomAudit.lean`). `waterfill_exclusion`: in the high-resolution regime
`KL(R)/KL(R') = 4^{−(R−R')}`. `converse_w15c` states the 0.5B measurements above as ℚ facts. Live tree:
`cd proof/corpus && lake build Perf.CostCorrespondence AxiomAudit` 9172 jobs, exit 0
(`/home/socol/oc-tmp/escalation/2026-10-11-converse-live.log`); 9/9 mutation controls (m13–m21) exit 1
(`/home/socol/oc-tmp/escalation/2026-10-11-converse-lean.log`). The size trend is not yet in Lean.

## Not covered (W15c)

- 1.5B and 3B on 8-layer subsets; 7B and above not measured (no layer-streamed capture yet).
- Shannon lower bound is an `anchor`; nonlinear dependence not probed; C4 calibration only.

# ONEBIT W15d: output-fitted transfer, first configuration

Command: `python tools/onebit/fold_transfer.py …` (sha256 `4032c1e27555ffb6…`), receipts
`/home/socol/oc-tmp/escalation/2026-10-11-fold-transfer{,-block}.{json,log}`. Qwen2.5-0.5B, all 24 blocks, one-bit
signs with per-row and per-column amplitudes at a uniform 1.133 b/w all-in, 100 C4 calibration chunks, 40 held-out.

The loop alternates the ridge solve (`src/learn/ridge.rs` `FoldTrainer::solve`, `W = (XᵀX + λI)⁻¹XᵀY` against the
teacher's outputs on the student's inputs) with one-bit GPTQ rounding and error feedback (`src/learn/onebit.rs`
`gptq_row`, `feedback_u`), then refits the amplitudes by their normal equations. `fold_transfer.py` is a Python
mirror; it matches a transcription of `gptq_row` to 1e-16.

| variant | held-out KL (nats/token) | top-1 agreement | wall |
|---|---:|---:|---:|
| RTN | 12.70 | 0.000 | 0.03 s |
| GPTQ one-bit, teacher Grams (W9 reproduction) | 14.86 | 0.000 | 31.6 s |
| GPTQ one-bit, student-upstream Grams | 7.08 | 0.047 | 38.0 s |
| ridge target + GPTQ | 5.25 | 0.087 | 44.9 s |
| + amplitude refits (layer target) | 3.81 | 0.206 | ~90 s |
| + block-output target | **3.09** | 0.261 | 88.8 s |
| control: ridge with full-precision weights | 0.0195 | 0.921 | 22.0 s |
| control: same, mutated | 7.09 | 0.032 | 22.2 s |

- Prediction `fold-transfer-block-1` confirmed (block KL in 1.5–3.5, control < 0.05, mutant > 5).
- Each output-fitting step cut KL; together 4.8× against GPTQ in 89 s. This first configuration sits at 154× Q4_K_M
  (≈ 0.020).
- Not yet in this configuration: rate water-filled across tensors (it spends a uniform 1.133 b/w where W15b puts
  `attn_v` and FFN far above `attn_q`), a KL-weighted (`G`) target instead of block-output MSE, enough calibration to
  stop overfitting (in-sample 0.86 against held-out 3.09), coding in the K-FAC eigenbasis, end-to-end refinement.

## 27B time (`model`, calibrated 88.7 s predicted against 90 s measured on 0.5B)

At 51,200 calibration tokens on the RTX 5080: the fit as written ran in f64 (0.85 TFLOPS) and models at about 16,300 s;
in fp32 about 1,200 s; fp32 with a fused column kernel about 736 s. Arithmetic in `/home/socol/oc-tmp/ft/b2.py`.
f64 is retired from this path: `src/learn/ridge.rs`, `src/learn/onebit.rs` and `tools/onebit/*.py` move to fp32.

## Not covered (W15d)

- 0.5B only; Python mirror, not the Rust path; C4 only; the 27B time is a model, not a run.

Board row: `onebit-exact-one-bit-and-FubKIC3prjKrgnryWWxrF1` (LEARN-ONEBIT, `docs/plan.md` row 248).
