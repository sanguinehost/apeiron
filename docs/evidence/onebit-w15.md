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
