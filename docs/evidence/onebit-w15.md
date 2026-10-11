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
