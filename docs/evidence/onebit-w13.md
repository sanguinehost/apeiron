# Onebit W13: `IQ1_S` and `had-gptq1` against the spectrum of the layer Gram matrix

Population: the 21 `ffn_down` tensors `blk.3`..`blk.23` of Qwen2.5-0.5B, Gram `H` from the C4 calibration
(`/home/socol/oc-tmp/gptq-w2/gram/`). Relative layer error `ρ = Σ_rows dHdᵀ / Σ_rows wHwᵀ` with `d` the arm's weight
error against f16 (`Perf.OnebitW10`: the layer error is `rᵀHr`). This is not the KLD population; `ρ` is a layer
quantity, KLD is the model quantity.

## Theory (`Perf/OnebitW13.lean`, `proved`, closed cone)

| statement | theorem |
|---|---|
| `rᵀ(RᵀR)r = Σ_i ((Rr)_i)²` | `quad_eq` |
| the Cholesky pivot is at most the diagonal, so error feedback at one quantizer noise never exceeds independent rounding | `pivot_le_diag`, `shaped_le_white` |
| `det H = ∏ R_ii²` and `det H ≤ ∏ H_ii` (Hadamard) | `det_eq_prod_pivots`, `hadamard_chol` |
| `n (∏ x_i)^{1/n} ≤ Σ x_i`: fed-back error `≥ n q² GM(λ)` against `n q² AM(λ)` for white noise | `gm_le_am`, `hi_res_floor` |
| the best amplitude on a unit source leaves `1 − m1²`; for the unit Gaussian (`m1² = 2/π`, anchor) `0.3633 < 1 − 2/π < 0.3634`, above the Gaussian rate–distortion value `1/4` at one bit | `sign_mse_min`, `sign_loss_bounds`, `sign_loss` |

Reverse water-filling for weight `λ_i` is the `model` grade floor: `D(θ) = Σ min(θ, s²λ_i)`,
`R(θ) = (1/2n) Σ log₂⁺(s²λ_i/θ)`, row model `w ~ N(0, s²I)`. `hi_res_floor` is the proved case with every direction
above the water level. The Cover & Thomas `R(D)` theorem is not retrieved here: `anchor`.

## Measurements

| item | command | receipt |
|---|---|---|
| per-arm `ρ`, `AM/GM` of `H` | `tools/onebit/arm_compare.py` (`/home/socol/Workspace/spinor-ttl/.venv/bin/python`, 14 s) | `/home/socol/oc-tmp/escalation/2026-10-11-arm-compare.json` |
| isotropic water-filling floor | `tools/onebit/waterfill.py` (17 s) | `/home/socol/oc-tmp/escalation/2026-10-11-waterfill.json` |
| instrument control | `tools/onebit/waterfill.py --self-test` | white spectrum gives `ρ = 2^{-2R}`; a two-level spectrum matches its closed form to 1e-9; a throwaway copy with `np.minimum → np.maximum` in `rho_of_theta` raises `AssertionError (0.5, 1.0)`, exit 1; copy removed |

### Arms (`measured`, `ρ` against f16 on `H`)

| arm | b/w | `ρ` blocks 4–20 | `ρ` blk 3 | `ρ` blk 21 | `ρ` blk 23 |
|---|---:|---|---:|---:|---:|
| Q4_K_M (imatrix) | 4.5–6.56 | 0.0002–0.0049 | 0.0001 | 0.0001 | 0.0001 |
| `IQ1_M` | 1.75 | 0.10–0.19 | 0.0153 | 0.0203 | 0.0243 |
| `IQ1_S` | 1.5625 | 0.13–0.23 | 0.0194 | 0.0255 | 0.0327 |
| `had-gptq1` | 1.125 | 0.10–0.22 | 0.0047 | 0.0097 | 0.0168 |
| Hadamard-128 sign, mean-abs amplitude (`had_rtn`) | 1.125 | 0.32–0.36 | 0.125 | 0.121 | 0.246 |
| plain RTN sign | 1.125 | 0.37–0.56 | 1.34 | 1.01 | 0.52 |

Instrument checks (computed from the receipt):

- `ln GM(eig H) ≤ ln GM(diag H)` holds on 21/21 tensors (Hadamard).
- The `had_rtn` median `ρ = 0.339` against the white-Gaussian sign value `1 − 2/π = 0.363`.
- `AM/GM(eig H)` is 2.9–6.6 dB on blocks 4–20 and 22, and 15.6 / 12.6 / 12.2 dB on blocks 3 / 21 / 23.
- `had_rtn → had-gptq1` realises 0.62–0.96 (mean 0.73) of the `AM/GM(eig)` gain in dB.
- Plain RTN is above the white-Gaussian sign value on all 21 tensors (0.37–1.34; peaks on blocks 3, 11, 21, 23): it
  is not white against `H`. Prediction `arm-shaping-1-1`: refuted with plain RTN as the white baseline; the
  same-basis baseline (`had_rtn`) gives the gain fractions above.

### Floor (`model`, isotropic Gaussian rows; `ρ` ten-thousandths in the Lean verdicts)

| block | `r99(H)` | floor `ρ` at 1.0 b/w | floor `ρ` at 1.125 | `had-gptq1` `ρ` | floor rate at Q4_K-type `ρ` (b/w) |
|---|---:|---:|---:|---:|---:|
| 3 | 1037 | 0.0038 | 0.0032 | 0.0047 | 3.84 |
| 21 | 2228 | 0.0099 | 0.0083 | 0.0097 | 4.73 |
| 23 | 2201 | 0.0111 | 0.0092 | 0.0168 | 4.55 |
| 4–20, 22 | 3798–4391 | 0.052–0.127 | 0.044–0.107 | 0.10–0.22 | 3.1–5.4 |

- `had-gptq1` sits 1.9–2.4× above its floor `ρ` at 1.125 b/w on blocks 4–20 and 22, and 1.5 / 1.2 / 1.8× on blocks
  3 / 21 / 23. The floor rate at the measured `had-gptq1` `ρ` is 0.54–0.67 b/w on blocks 4–20 and 22 (rate gap
  0.46–0.59 b/w against 1.125), and 0.85 / 1.02 / 0.74 on blocks 3 / 21 / 23 (gap 0.28 / 0.10 / 0.39).
- `IQ1_S` sits 3.8–6.6× above its floor `ρ` at 1.5625 b/w (block 3: 11.2×).
- Q4_K spends 4.5 b/w where the floor is 3.1–3.4 b/w on the Q4_K-type blocks.
- At 1 b/w the isotropic floor reaches the Q4_K-type level (0.0045) on block 3 only.

Prediction `floor-1`: refuted on two clauses (block 3 arm above the floor, not below; floor rate at Q4_K `ρ` is
3.1–5.4 b/w, not 1.3), confirmed on the typical-block ratio. Verdicts: `arm_above_floor_1125`,
`one_bit_floor_vs_q4k`, `floor_rate_at_q4k`.

## What this says

- The Gerzon–Craven gain `AM/GM(eig H)` is what separates the arms: `had-gptq1` realises most of it, and `IQ1_S`
  (diagonal proxy of `H`) reaches `GM(diag H)`, which equals `GM(eig H)` only when `H` is diagonal.
- Under the isotropic row model, 1 b/w at the Q4_K-type error is reachable on block 3 and unreachable on the typical
  blocks (floor `ρ` 0.07–0.13 at 1 b/w). The model is `model` grade: a weight row with structure (low effective rank,
  heavy tails) can sit below it. The measured `had-gptq1` `ρ` is within 1.2–2.4× of the floor everywhere.
- The remaining rate gap of 0.46–0.59 b/w on the typical blocks has two measured parts: the sign quantizer, whose
  one-bit loss `1 − 2/π` exceeds the Gaussian `1/4` (`sign_loss`), and the spectral shaping not realised (0.62–0.96
  of the `AM/GM` gain; mean 0.73). Their split is not measured here.

## Not covered

rustc/CUDA codegen of the instruments; the population beyond one model and one calibration set; the map from layer
`ρ` to KLD (measured separately in `docs/evidence/onebit-w9.md`, `onebit-w12.md`); the Gaussian row model; the
Cover & Thomas `R(D)` theorem (anchor, not retrieved).
