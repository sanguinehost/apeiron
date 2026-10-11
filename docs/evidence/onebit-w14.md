# Onebit W14: sampled-Fisher stiffness of the five arms against P1 KLD

Population: the 21 `ffn_down` tensors `blk.3`..`blk.23` of Qwen2.5-0.5B (P1, 91,521,024 weights), C4 calibration
(`/home/socol/oc-tmp/seal/calib-c4.txt`), 100 chunks of 512 tokens, 4 sampled targets per chunk, 400 samples.

## Quantity

For a sampled target sequence `s ~ p(·|x)` (true Fisher, not the empirical Fisher) the score of the weight `W_l` is
`J_l = ∂(−log p(s))/∂W_l = Σ_t g_t x_tᵀ`. A weight error `D_l = W_arm − W_f16` has second-order KL
`½ E[⟨J_l, D_l⟩²]` (Gauss-Newton, `model` grade beyond the small-perturbation regime). No Kronecker factorisation:
`⟨J, D⟩` is computed per sample from the full `J`. With all 21 blocks perturbed, the joint is
`½ E[(Σ_l ⟨J_l, D_l⟩)²]`; the additivity ratio is joint over `Σ_l` single-block.

## Measurement (`measured`)

Command: `fisher_capture.py` (`/home/socol/Workspace/spinor-ttl/.venv/bin/python`, args `…-stiffness.json 100 4`), 87 s,
under the lane lock and a 24 G cap. Receipt: `/home/socol/oc-tmp/escalation/2026-10-11-stiffness.json` (+ `.scores.npy`),
log `…-stiffness.log`. Prediction `stiff-1-1`.

| arm | predicted Σ_l (nats/token) | predicted joint | joint SEM | P1 KLD | P1 / predicted (Σ_l) | joint / Σ_l |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 0.0039 | 0.0040 | 0.0003 | 0.0243 (whole model) | 6.30 | 1.03 |
| `IQ1_M` | 0.3446 | 0.3299 | 0.0211 | 0.5191 | 1.51 | 0.96 |
| `IQ1_S` | 0.4564 | 0.4573 | 0.0288 | 0.6315 | 1.38 | 1.00 |
| `had-gptq1` | 0.3735 | 0.2919 | 0.0220 | 0.8345 | 2.23 | 0.78 |
| RTN sign | 6.7661 | 3.5530 | 0.2538 | 10.3846 | 1.53 | 0.53 |

Predicted order (Σ_l): Q4_K_M < `IQ1_M` < `had-gptq1` < `IQ1_S` < RTN. Measured P1 order: Q4_K_M < `IQ1_M` < `IQ1_S` <
`had-gptq1` < RTN. Spearman 0.9, one adjacent inversion.

## Verdict on `stiff-1-1`

- Rank clause (Spearman +1): refuted. `had-gptq1` is predicted 0.08 nats/token below `IQ1_S` (Σ_l; the per-arm joint SEM
  is 0.02–0.03) and measured 0.20 above it.
- Ratio clause (predicted / P1 ∈ [0.3, 3] on `IQ1_M`, `IQ1_S`, `had-gptq1`): confirmed on Σ_l (0.66, 0.72, 0.45) and on
  the `had-gptq1` joint (0.35).
- Q4_K_M: the P1 figure is a whole-model file, so only its order is claimed (first in both).
- Reading: the quadratic form with the full Fisher puts `had-gptq1` and `IQ1_S` at the same scale as measured KLD but
  does not order them. `had-gptq1` is under-predicted most (P1/predicted 2.23, 2.86 on the joint). That is the
  signature of a perturbation outside the small-error regime, or a term the quadratic form does not carry; which is
  not determined here.
- Additivity (`model` reading of the face falsifier): joint/Σ_l is 1.00 `IQ1_S`, 0.96 `IQ1_M`, 1.03 Q4_K_M, 0.78
  `had-gptq1`, 0.53 RTN. The arms with the largest per-weight error are sub-additive, so cross-layer score terms are
  negatively correlated there. The mapping from this ratio to hyperbolic / parabolic / elliptic face of the depth
  operator is a prediction made before the run, not a derived identity (`anchor`).

Lean verdicts (`proved`, closed cone, `Perf/OnebitW14.lean`): `pred_order`, `p1_order`, `rank_inversion`,
`ratio_band`, `additivity`.

## Not covered

- The estimator is first-order in the score: a second-order term of `D` (third derivative of the KL) is not computed.
- No mutation control of `fisher_capture.py` has been run; the check that it is non-vacuous here is that the five arms
  land within 0.66–6.3× of independent KLD measurements.
- One model, one calibration set, 400 samples; CUDA/torch codegen; the `had-gptq1` fit set versus the calibration set
  is not compared (in-sample question from the W13 review stays open).
- Single-layer-only quantised KLD runs (needed to read the depth face from KLD directly) are not done.
